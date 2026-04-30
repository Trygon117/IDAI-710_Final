import torch
import torch.nn as nn


class AttentionMLP(nn.Module):
    """Computes a raw attention score for each (node, hyperedge) pair.

    We use CNN embeddings z instead of GNN hidden states so the attention
    stays tied to image content and doesn't get tangled up with graph structure.
    """

    def __init__(self, z_dim=128, e_dim=64, hidden=128, num_hyperedges=2):
        super().__init__()
        # each hyperedge gets its own learned embedding
        self.e_embed = nn.Embedding(num_hyperedges, e_dim)
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + e_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, edge_idx):
        """
        z:        (N, z_dim) CNN embeddings
        edge_idx: int — which hyperedge we're scoring
        Returns:  (N, 1) raw attention logits
        """
        idx = torch.tensor(edge_idx, device=z.device)
        e = self.e_embed(idx).unsqueeze(0).expand(z.size(0), -1)  # broadcast edge emb to all nodes
        return self.mlp(torch.cat([z, e], dim=-1))


# old name kept around in case anything imports it
HyperedgeAttentionMLP = AttentionMLP
