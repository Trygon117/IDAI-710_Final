"""Stage 1 training helpers — called by scripts/train_stage1.py."""
import torch
import torch.nn as nn
from awhgcn.eval.metrics import compute_metrics


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for batch in loader:
        x = batch["volume"].to(device)
        y = batch["label"].to(device)
        _, logits = model(x)  # we only need logits here, not the embedding
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, preds, labs = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["volume"].to(device)
            y = batch["label"].to(device)
            _, logits = model(x)
            total += criterion(logits, y).item()
            preds.extend(logits.argmax(-1).cpu().numpy())
            labs.extend(y.cpu().numpy())
    m = compute_metrics(preds, labs)
    m["loss"] = total / len(loader)
    return m
