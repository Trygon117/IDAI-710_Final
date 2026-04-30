#!/usr/bin/env python3
"""
Generate a 6x6 soft confusion matrix heatmap showing average predicted probabilities
broken down by true fine-grained disease subtype.
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from awhgcn.data.cohort import CLASS_NAMES

CKPT_DIR = ROOT / "data" / "checkpoints"
PARQUET = ROOT / "data" / "manifests" / "patients_300.parquet"
TABULAR_CSV = ROOT / "data" / "raw" / "tabular" / "investigator_nacc72.csv"
OUT_PATH = ROOT / "results" / "figures" / "soft_prob_heatmap.png"

# maps cohort group key; display name matching CLASS_NAMES order
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
    emb  = state["embeddings"]
    w = state["model_state"]["classifier.weight"]
    b = state["model_state"]["classifier.bias"]
    logits = emb @ w.T + b
    return F.softmax(logits, dim=-1).numpy()   # (N, 6)


def plot_soft_confusion_matrix(true_labels, predicted_probs, class_names,
                                title="Soft Probability Heatmap"):
    num_classes = len(class_names)
    soft_cm = np.zeros((num_classes, num_classes))

    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)

    for i, label in enumerate(class_names):
        class_indices = np.where(true_labels == label)[0]
        if len(class_indices) > 0:
            class_probs = predicted_probs[class_indices]
            soft_cm[i, :] = np.mean(class_probs, axis=0)

    brand_purple = "#7B2D8B"
    custom_cmap = sns.light_palette(brand_purple, as_cmap=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        soft_cm,
        annot=True,
        fmt=".1%",
        cmap=custom_cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.8,
        linecolor="white",
    )
    plt.ylabel("True Disease")
    plt.xlabel("Average Predicted Probability")
    plt.title(title)
    plt.tight_layout()


def main():
    cohort = pd.read_parquet(PARQUET)
    naccids = cohort["NACCID"].tolist()
    groups = cohort["group"].tolist()
    print(f"Cohort: {len(naccids)} patients")

    # map each patient's group key to its display name
    display_labels = [GROUP_TO_DISPLAY.get(g, g) for g in groups]

    all_probs = []
    for fold in range(5):
        ckpt = CKPT_DIR / f"stage2_fold{fold}.pt"
        if ckpt.exists():
            all_probs.append(compute_fold_probs(ckpt))
            print(f"  Loaded fold {fold}")
    if not all_probs:
        raise FileNotFoundError(f"No stage2 checkpoints found in {CKPT_DIR}")
    probs = np.mean(np.stack(all_probs, axis=0), axis=0)   # (N, 6)

    plot_soft_confusion_matrix(
        true_labels = display_labels,
        predicted_probs = probs,
        class_names = CLASS_NAMES,
        title = "AWHGCN Soft Confusion Matrix\n(5-Fold CV Average)",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
