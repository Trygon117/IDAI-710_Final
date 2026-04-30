import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from .metrics import compute_metrics


def _lr_pipe():
    # always scale before LR — tabular features have very different ranges
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def run_lr_baseline(X_tr, y_tr, X_v, y_v):
    pipe = _lr_pipe().fit(X_tr, y_tr)
    return compute_metrics(pipe.predict(X_v), y_v)


def run_xgb_baseline(X_tr, y_tr, X_v, y_v):
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0,
    ).fit(X_tr, y_tr)
    return compute_metrics(clf.predict(X_v), y_v)


def run_cnn_baseline(embeddings, labels, train_idx, val_idx):
    """Linear probe on top of frozen Stage 1 CNN embeddings."""
    emb = np.asarray(embeddings)
    X_tr, y_tr = emb[train_idx], np.asarray(labels)[train_idx]
    X_v,  y_v  = emb[val_idx],  np.asarray(labels)[val_idx]
    pipe = _lr_pipe().fit(X_tr, y_tr)
    return compute_metrics(pipe.predict(X_v), y_v)
