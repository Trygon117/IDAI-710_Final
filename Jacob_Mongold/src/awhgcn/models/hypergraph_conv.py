import torch
import torch.nn as nn
from .attention import AttentionMLP


class AWHGConv(nn.Module):
    """Attention-Weighted Hypergraph Convolution layer.

    Implements: X' = D_v^{-1} (B ⊙ α) (B ⊙ α)^T X W

    Group C nodes have B[i,0]=B[i,1]=0 so they don't contribute to or receive from
    the diagnosis hyperedges — but we still read out their alpha_raw[:, 0:1] values
    to get P(AD | i) at inference time.
    """

    def __init__(self, in_channels, out_channels, num_hyperedges=2,
                 z_dim=128, dropout=0.3):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.attn = AttentionMLP(z_dim=z_dim, num_hyperedges=num_hyperedges)
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, B, z):
        """
        x: (N, in_channels) node features
        B: (N, E) incidence matrix — 1 if node i is in hyperedge e
        z: (N, z_dim) CNN embeddings used for attention scoring
        Returns:
          x_out:     (N, out_channels)
          alpha_raw: (N, E) raw logits — cols 0/1 used for Group C P(AD) readout
        """
        N, E = B.shape

        # step 1: get raw attention score for every (node, edge) pair
        alpha_raw = torch.cat(
            [self.attn(z, e) for e in range(E)], dim=1
        )  # (N, E)

        # step 2: softmax across nodes per edge, but only over members (mask non-members to -inf)
        alpha_norm = alpha_raw.masked_fill(B == 0, float("-inf"))
        alpha_norm = alpha_norm.softmax(dim=0)
        alpha_norm = torch.nan_to_num(alpha_norm, nan=0.0)  # empty hyperedge → 0

        # step 3: weight the incidence matrix by attention
        B_w = B * alpha_norm  # (N, E)

        # step 4: standard hypergraph convolution with the weighted incidence matrix
        xW = self.W(x)           # (N, out_channels)
        H_e = B_w.t() @ xW      # aggregate node features into each hyperedge
        x_agg = B_w @ H_e        # scatter back to nodes
        D_v = B_w.sum(dim=1).clamp(min=1e-6)
        x_out = x_agg / D_v.unsqueeze(-1)  # normalize by node degree

        x_out = self.norm(self.drop(x_out))
        return x_out, alpha_raw
