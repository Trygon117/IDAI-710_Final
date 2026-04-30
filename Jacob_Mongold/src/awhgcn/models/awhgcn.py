import torch
import torch.nn as nn
from .hypergraph_conv import AWHGConv


class AWHGCN(nn.Module):
    """Full model: CNN backbone + attention-weighted hypergraph convolution."""

    def __init__(self, cnn_backbone, tabular_dim, n_edges, cfg):
        super().__init__()
        self.cnn = cnn_backbone       # accessed by training script as model.cnn
        self.backbone = cnn_backbone  # alias so both names work

        z_dim = cfg.cnn.proj_dim      # CNN output dim fed into attention

        # project tabular features up to 128-d before concatenating with image embedding
        self.tabular_proj = nn.Linear(tabular_dim, 128)
        self.node_proj = nn.Linear(z_dim + 128, cfg.hypergraph.hidden_dim)

        self.conv_layers = nn.ModuleList([
            AWHGConv(
                in_channels=cfg.hypergraph.hidden_dim,
                out_channels=cfg.hypergraph.hidden_dim,
                num_hyperedges=n_edges,
                z_dim=z_dim,
                dropout=cfg.hypergraph.dropout,
            )
            for _ in range(cfg.hypergraph.n_conv_layers)
        ])

        num_classes = getattr(cfg.model, "num_classes", 6)
        self.classifier = nn.Linear(cfg.hypergraph.hidden_dim, num_classes)
        self.drop = nn.Dropout(cfg.hypergraph.dropout)
        self._zero_cnn = getattr(cfg.cnn, "zero_embeddings", False)

    def forward(self, volumes_or_z, tabular, H, use_precomputed_z=False):
        """
        volumes_or_z: (N, 1, 96, 96, 96) raw volumes OR (N, z_dim) precomputed embeddings
        tabular:      (N, tabular_dim)
        H:            (N, E) incidence matrix
        Returns: logits (N, num_classes), node_embeddings (N, hidden_dim), alpha_raw (N, E)
        """
        if use_precomputed_z:
            z = volumes_or_z
        else:
            z = self.backbone.encode(volumes_or_z)

        if self._zero_cnn:
            z = torch.zeros_like(z)

        tabular_emb = torch.relu(self.tabular_proj(tabular))
        x = torch.relu(self.node_proj(torch.cat([z, tabular_emb], dim=-1)))

        alpha_raw = None
        for conv in self.conv_layers:
            x, alpha_raw = conv(x, H, z)
            x = torch.relu(x)

        logits = self.classifier(self.drop(x))
        return logits, x, alpha_raw
