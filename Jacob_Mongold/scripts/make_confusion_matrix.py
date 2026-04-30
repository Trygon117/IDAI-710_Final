#!/usr/bin/env python3
"""
Generate a 6x6 confusion matrix heatmap for the AWHGCN 6-class classifier.

Hard predictions (argmax of classifier logits) are averaged across all 5 CV folds.
Rows = True disease subtype, Cols = Predicted disease subtype.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from awhgcn.data.cohort import CLASS_NAMES, LABEL_MAP

CKPT_DIR = ROOT / "data" / "checkpoints"
PARQUET = ROOT / "data" / "manifests" / "patients_300.parquet"
OUT_PATH = ROOT / "results" / "figures" / "confusion_matrix.png"

GROUP_TO_DISPLAY = {
    "Normal": "Normal",
    "MCI": "MCI",
    "AD": "Alzheimer's",
    "Vascular": "Vascular",
    "LewyBody": "Lewy Body",
    "FTD": "Frontotemporal",
}


def compute_fold_probs(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    emb = state["embeddings"]
    w = state["model_state"]["classifier.weight"]
    b = state["model_state"]["classifier.bias"]
    logits = emb @ w.T + b
    return F.softmax(logits, dim=-1).numpy()   # (N, 6)


def main():
    cohort = pd.read_parquet(PARQUET)
    naccids = cohort["NACCID"].tolist()
    groups = cohort["group"].tolist()
    print(f"Cohort: {len(naccids)} patients")

    true_labels = np.array([LABEL_MAP[g] for g in groups])

    all_probs = []
    for fold in range(5):
        ckpt = CKPT_DIR / f"stage2_fold{fold}.pt"
        if ckpt.exists():
            all_probs.append(compute_fold_probs(ckpt))
            print(f"  Loaded fold {fold}")
    if not all_probs:
        raise FileNotFoundError(f"No stage2 checkpoints found in {CKPT_DIR}")

    probs_avg = np.mean(np.stack(all_probs, axis=0), axis=0)   # (N, 6)
    preds = probs_avg.argmax(axis=-1)                       # 0-5

    print("\n--- 6-Class Classification Report ---")
    print(classification_report(true_labels, preds, target_names=CLASS_NAMES, zero_division=0))

    # build percentage confusion matrix
    n = len(CLASS_NAMES)
    counts = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, preds):
        counts[t, p] += 1
    row_totals = counts.sum(axis=1, keepdims=True)
    pct = np.where(row_totals > 0, counts / row_totals * 100.0, 0.0)

    brand_purple = "#7B2D8B"
    custom_cmap = sns.light_palette(brand_purple, as_cmap=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pct,
        annot=True,
        fmt=".1f",
        cmap=custom_cmap,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.8,
        linecolor="white",
        vmin=0,
        vmax=100,
    )
    plt.ylabel("True Disease")
    plt.xlabel("Predicted Disease")
    plt.title("AWHGCN Confusion Matrix\n(5-Fold CV Average, % of Row)")
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
