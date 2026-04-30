"""Graph construction and encoding utilities for Stage 2."""
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_incidence_matrix(df, tabular_tensor, train_class_indices, k=5):
    """Build H (N, n_classes+N): first n_classes cols are diagnosis hyperedges, rest are kNN.

    train_class_indices: dict mapping group name → list of train-set node indices
                         (val nodes excluded to prevent data leakage)
    """
    N = len(df)
    n_classes = len(train_class_indices)
    X = tabular_tensor.numpy()

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    _, indices = nbrs.kneighbors(X)  # (N, k+1) including self

    H = torch.zeros(N, n_classes + N, dtype=torch.float32)

    # one diagnosis hyperedge per class — only train nodes are members
    for cls_col, node_indices in enumerate(train_class_indices.values()):
        for i in node_indices:
            H[i, cls_col] = 1.0

    # kNN hyperedges: patient i's hyperedge contains its k nearest neighbors
    for i in range(N):
        for j in indices[i]:
            H[j, n_classes + i] = 1.0

    return H


def encode_with_grad(cnn, dataset, device, batch_size=4):
    """Encode all patients through the CNN with gradient checkpointing."""
    import torch.utils.checkpoint as ckpt
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    zs = []
    for batch in loader:
        x = batch["volume"].to(device)
        z = ckpt.checkpoint(cnn.encode, x, use_reentrant=False)
        zs.append(z)
    return torch.cat(zs, dim=0)


def encode_no_grad(cnn, dataset, device, batch_size=4):
    """Encode all patients without gradients — used for validation and inference."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cnn.eval()
    zs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["volume"].to(device)
            zs.append(cnn.encode(x).cpu())
    return torch.cat(zs, dim=0)
