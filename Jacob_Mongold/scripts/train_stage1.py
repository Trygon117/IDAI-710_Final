#!/usr/bin/env python3
"""Stage 1: Warm up the CNN backbone on labeled patients (Groups A + B) before joint training."""

import os
os.environ.setdefault("PANDAS_FUTURE_INFER_STRING_CONVERT_BACKEND", "python")

# pandas/pyarrow must be imported before torch on Windows to avoid a DLL conflict segfault
import pandas as _pd  # noqa: F401

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from awhgcn.data.cohort import load_cohort, LABEL_MAP
from awhgcn.data.dataset import NACCDataset
from awhgcn.data.preprocessing import get_aug_transforms
from awhgcn.models.cnn_backbone import CNNBackbone
from awhgcn.training.stage1_cnn import train_epoch, eval_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--debug", action="store_true", help="3 epochs, 10 patients")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.epochs:
        cfg.stage1.epochs = args.epochs

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, used_cols = load_cohort(
        cfg.cohort.tabular_csv, mri_csv=cfg.cohort.mri_csv,
        n_per_group=cfg.cohort.n_per_group, seed=cfg.seed,
    )

    # Stage 1 trains on all 6 labeled groups
    preprocessed_dir = Path(cfg.paths.preprocessed_dir)
    labeled = df[df["group"].isin(LABEL_MAP.keys())].copy()
    labeled = labeled[
        labeled["NACCID"].apply(lambda x: (preprocessed_dir / f"{x}.pt").exists())
    ].reset_index(drop=True)
    print(f"Labeled patients with preprocessed MRI: {len(labeled)}")

    if len(labeled) == 0:
        print("No preprocessed MRI found — run scripts/preprocess.py first.")
        return

    if args.debug:
        labeled = labeled.head(10)
        cfg.stage1.epochs = 3

    # stratified split so val set has the same class balance as train
    train_idx, val_idx = train_test_split(
        range(len(labeled)), test_size=0.2,
        stratify=labeled["label"].values, random_state=cfg.seed,
    )
    train_ds = NACCDataset(labeled.iloc[train_idx], preprocessed_dir, used_cols,
                           transform=get_aug_transforms())
    val_ds   = NACCDataset(labeled.iloc[val_idx],   preprocessed_dir, used_cols,
                           transform=None)

    train_dl = DataLoader(train_ds, batch_size=cfg.stage1.batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.stage1.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)

    num_classes = cfg.model.num_classes
    model = CNNBackbone(cfg.cnn.out_dim, cfg.cnn.proj_dim, num_classes, cfg.cnn.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.stage1.lr, weight_decay=cfg.stage1.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.stage1.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.stage1.label_smoothing)

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(cfg.stage1.epochs):
        tr_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        val_m   = eval_epoch(model, val_dl, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d} | train_loss={tr_loss:.4f} "
              f"| val_loss={val_m['loss']:.4f} | val_f1={val_m['macro_f1']:.4f} "
              f"| val_acc={val_m['accuracy']:.4f}")

        # save whenever we beat the best val F1
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            torch.save(model.state_dict(), ckpt_dir / "stage1_best.pt")
            print(f"  Saved best checkpoint (val_f1={best_f1:.4f})")

    print(f"\nStage 1 complete. Best val macro-F1: {best_f1:.4f}")
    print(f"Checkpoint -> {ckpt_dir}/stage1_best.pt")


if __name__ == "__main__":
    main()
