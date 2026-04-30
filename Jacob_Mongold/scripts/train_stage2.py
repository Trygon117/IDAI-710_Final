#!/usr/bin/env python3
"""Stage 2: Joint CNN + Hypergraph training (transductive, all 300 nodes)."""

import os
os.environ.setdefault("PANDAS_FUTURE_INFER_STRING_CONVERT_BACKEND", "python")

# pandas/pyarrow must be imported before torch on Windows — CUDA DLLs conflict
# with pyarrow's DLLs if they load in the wrong order, causing a silent segfault.
import pandas as _pd  # noqa: F401

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from awhgcn.data.cohort import load_cohort, get_class_indices, get_tabular_tensor
from awhgcn.data.dataset import NACCDataset
from awhgcn.models.cnn_backbone import CNNBackbone
from awhgcn.models.awhgcn import AWHGCN
from awhgcn.training.losses import combined_loss
from awhgcn.training.cv import get_cv_splits
from awhgcn.training.stage2_joint import build_incidence_matrix, encode_with_grad, encode_no_grad
from awhgcn.eval.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--single-fold", action="store_true")
    p.add_argument("--fold-idx", type=int, default=None,
                   help="Run only this fold index (implies --single-fold)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--conservative-lr", action="store_true",
                   help="Drop CNN LR to 1e-6 if Stage 2 is unstable")
    return p.parse_args()


def run_fold(fold_idx, train_idx, val_idx, group_c_idx,
             df, used_cols, tabular, full_dataset,
             cfg, device, ckpt_dir):

    print(f"\n=== Fold {fold_idx} ===")

    class_indices = get_class_indices(df)

    # only put train nodes in the diagnosis hyperedges — val nodes excluded to prevent data leakage
    train_idx_set = set(train_idx)
    train_class_indices = {
        grp: [i for i in indices if i in train_idx_set]
        for grp, indices in class_indices.items()
    }
    H = build_incidence_matrix(df, tabular, train_class_indices, k=cfg.hypergraph.k).to(device)
    tabular_dev = tabular.to(device)

    # load Stage 1 weights as starting point — much better than training from scratch
    cnn = CNNBackbone(cfg.cnn.out_dim, cfg.cnn.proj_dim, cfg.model.num_classes, cfg.cnn.dropout).to(device)
    stage1_ckpt = ckpt_dir / "stage1_best.pt"
    if stage1_ckpt.exists():
        cnn.load_state_dict(torch.load(stage1_ckpt, map_location=device, weights_only=True))
        print(f"  Loaded Stage 1 weights from {stage1_ckpt}")
        cnn.freeze_blocks(cfg.cnn.freeze_blocks)
    else:
        print("  WARNING: No Stage 1 checkpoint found — training from scratch.")

    n_edges = H.shape[1]
    model = AWHGCN(cnn, len(used_cols), n_edges, cfg).to(device)

    # CNN gets a much lower LR than the GNN parts — don't want to destroy the features we learned in Stage 1
    cnn_params = [p for p in model.cnn.parameters() if p.requires_grad]
    gnn_params = [p for name, p in model.named_parameters()
                  if not name.startswith("cnn.") and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": cnn_params, "lr": cfg.stage2.lr_cnn},
        {"params": gnn_params, "lr": cfg.stage2.lr_gnn},
    ], weight_decay=cfg.stage2.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.stage2.epochs,
    )

    labels = torch.tensor(df["label"].values, dtype=torch.long).to(device)
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    for i in train_idx:
        train_mask[i] = True
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    for i in val_idx:
        val_mask[i] = True

    best_f1, patience = 0.0, 0

    for epoch in range(cfg.stage2.epochs):
        model.train()
        # encode the full cohort in mini-batches with gradient checkpointing to save memory
        z_all = encode_with_grad(model.cnn, full_dataset, device,
                                 batch_size=cfg.stage2.batch_size_cnn)
        z_all = z_all.to(device)

        logits, embeddings, attn_raw = model(z_all, tabular_dev, H, use_precomputed_z=True)
        loss = combined_loss(logits, embeddings, labels, train_mask,
                             lambda_supcon=cfg.stage2.lambda_supcon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # validation pass — no gradients needed
        model.eval()
        with torch.no_grad():
            z_val = encode_no_grad(model.cnn, full_dataset, device,
                                   batch_size=cfg.stage2.batch_size_cnn).to(device)
            logits_v, emb_v, attn_v = model(z_val, tabular_dev, H, use_precomputed_z=True)

        val_preds  = logits_v[val_mask].argmax(-1).cpu().numpy()
        val_labels = labels[val_mask].cpu().numpy()
        metrics = compute_metrics(val_preds, val_labels)

        print(f"  Ep {epoch+1:3d} | loss={loss.item():.4f} | "
              f"val_f1={metrics['macro_f1']:.4f} | val_acc={metrics['accuracy']:.4f}")

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            patience = 0
            # save embeddings and attention weights alongside the model
            torch.save({
                "model_state": model.state_dict(),
                "embeddings": emb_v.cpu(),
                "attn_raw": attn_v.cpu() if attn_v is not None else None,
                "fold": fold_idx, "epoch": epoch + 1,
            }, ckpt_dir / f"stage2_fold{fold_idx}.pt")
        else:
            patience += 1
            if patience >= cfg.stage2.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"Fold {fold_idx}: best val macro-F1 = {best_f1:.4f}")
    return best_f1


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.epochs:
        cfg.stage2.epochs = args.epochs
    if args.conservative_lr:
        cfg.stage2.lr_cnn = 1e-6

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading cohort...", flush=True)
    df, used_cols = load_cohort(
        cfg.cohort.tabular_csv, mri_csv=cfg.cohort.mri_csv,
        n_per_group=cfg.cohort.n_per_group, seed=cfg.seed,
    )
    print(f"Cohort loaded: {len(df)} rows", flush=True)

    # only keep patients who actually have a preprocessed scan — can't train on zeros
    preprocessed_dir = Path(cfg.paths.preprocessed_dir)
    df = df[df["NACCID"].apply(
        lambda x: (preprocessed_dir / f"{x}.pt").exists()
    )].reset_index(drop=True)
    print(f"Patients with preprocessed MRI: {len(df)}", flush=True)

    tabular = get_tabular_tensor(df, used_cols)
    full_dataset = NACCDataset(df, preprocessed_dir, used_cols, transform=None)

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    splits = get_cv_splits(df, n_splits=5, seed=cfg.seed)
    if args.fold_idx is not None:
        splits = [splits[args.fold_idx]]
    elif args.single_fold:
        splits = splits[:1]

    fold_f1s = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        f1 = run_fold(fold_idx, train_idx, val_idx, [],
                      df, used_cols, tabular, full_dataset,
                      cfg, device, ckpt_dir)
        fold_f1s.append(f1)

    print(f"\nAll folds complete. Mean val macro-F1: {sum(fold_f1s)/len(fold_f1s):.4f}")


if __name__ == "__main__":
    try:
        main()
    except (Exception, SystemExit) as e:
        import traceback
        traceback.print_exc()
        print(f"\nCaught exit: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
