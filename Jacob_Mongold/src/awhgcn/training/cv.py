import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_cv_splits(df, n_splits=5, seed=42):
    """
    Returns list of (train_idx, val_idx) stratified across all 6 labeled classes.
    All nodes are labeled in the 6-class setting.
    """
    all_idx = list(range(len(df)))
    labels  = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_rel, val_rel in skf.split(all_idx, labels):
        splits.append((list(train_rel), list(val_rel)))

    return splits
