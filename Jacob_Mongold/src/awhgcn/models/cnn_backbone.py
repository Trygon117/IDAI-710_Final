import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121


class CNNBackbone(nn.Module):
    # DenseNet121 works well for 3D medical imaging and MONAI has a pretrained-friendly 3D version
    def __init__(self, out_dim=512, proj_dim=128, n_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=out_dim,
        )
        # projector maps to a lower-dim space for contrastive learning in Stage 1
        self.projector = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, proj_dim),
        )
        self.classifier = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        """Returns (z, logits) where z is the proj_dim-dim embedding."""
        z_raw = self.encoder(x)
        z = self.projector(z_raw)
        return z, self.classifier(z)

    def encode(self, x):
        """Just the embedding — used with gradient checkpointing in Stage 2."""
        return self.projector(self.encoder(x))

    def freeze_blocks(self, n=3):
        """Freeze the first n dense blocks so Stage 2 doesn't destroy Stage 1 features."""
        prefixes = [
            "features.denseblock1", "features.transition1",
            "features.denseblock2", "features.transition2",
            "features.denseblock3", "features.transition3",
            "features.denseblock4",
        ]
        to_freeze = set(prefixes[: n * 2])
        for name, param in self.encoder.named_parameters():
            if any(name.startswith(p) for p in to_freeze):
                param.requires_grad_(False)
