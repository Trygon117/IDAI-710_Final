import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020).

    Pulls same-class embeddings together and pushes different-class ones apart.
    Works better than cross-entropy alone when classes are similar (like AD vs non-AD dementia).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: (N, D) — L2-normalized inside here
        labels:     (N,)   integer class labels
        """
        z = F.normalize(embeddings, dim=-1)
        N = z.shape[0]

        sim = torch.matmul(z, z.T) / self.temperature       # cosine similarity scaled by temp

        eye = torch.eye(N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, float("-inf"))            # don't let a sample be its own positive

        pos_mask = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye

        log_prob = F.log_softmax(sim, dim=-1)
        n_pos = pos_mask.sum(dim=-1).clamp(min=1)
        # torch.where avoids -inf * 0 = NaN (IEEE 754 thing that bites you here)
        log_pos = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
        loss = -log_pos.sum(dim=-1) / n_pos
        return loss.mean()


def combined_loss(logits, embeddings, labels, mask, lambda_supcon=0.1, class_weights=None):
    """Cross-entropy on labeled nodes + SupCon on their embeddings."""
    ce = F.cross_entropy(logits[mask], labels[mask],
                         weight=class_weights, label_smoothing=0.1)
    if lambda_supcon > 0 and mask.sum() > 1:
        sc = SupConLoss()(embeddings[mask], labels[mask])
        return ce + lambda_supcon * sc
    return ce
