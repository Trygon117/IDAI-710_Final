import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)


def compute_metrics(preds, labels):
    # macro F1 is the main metric we care about since classes are balanced by design
    preds, labels = np.asarray(preds), np.asarray(labels)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0,
    )
    cm = confusion_matrix(labels, preds)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }
