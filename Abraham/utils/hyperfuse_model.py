import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.spatial_extractor = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pos_encoder = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        
        self.patch_pool = nn.AdaptiveAvgPool3d((3, 3, 3))

    def forward(self, x):
        # Extract the high-resolution 3D grid
        spatial_features = self.spatial_extractor(x)
        
        # Inject spatial awareness before pooling
        pos_features = spatial_features + self.pos_encoder(spatial_features)
        
        # Compress a copy of it into 27 macro-regions for the clinical hypergraph
        pooled_features = self.patch_pool(pos_features)
        
        N, C, D, H, W = pooled_features.size()
        
        # The sequence length is now D*H*W = 27
        patches = pooled_features.view(N, C, D * H * W).transpose(1, 2)
        
        # Return both: The 3D grid for the CNN residual, and the 1D patches for the population graph
        return spatial_features, patches
    
class ClinicalEdgeAttention(nn.Module):
    def __init__(self, clinical_features, hidden_dim):
        super().__init__()
        
        # Project the raw clinical data into a richer embedding
        self.feature_extractor = nn.Sequential(
            nn.Linear(clinical_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Create the mechanisms to ask questions and provide answers
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.hidden_dim = hidden_dim

        # Edge dropout to prevent memorizing specific patient connections
        self.attn_drop = nn.Dropout(p=0.1)

    def forward(self, clinical_data):
        # clinical_data shape: [N, clinical_features] where N is the number of patients
        
        # Upgrade the raw numbers into rich feature vectors
        h = self.feature_extractor(clinical_data) # h shape: [N, hidden_dim]
        
        # Generate the Queries and Keys for the patient population
        Q = self.q_linear(h) 
        K = self.k_linear(h) 
        
        # Calculate how much each patient relates to every other patient
        scores = torch.matmul(Q, K.transpose(0, 1)) # [N, hidden_dim] x [hidden_dim, N] = [N, N]
        
        # Scale the scores to keep the math stable during training
        scores = scores / math.sqrt(self.hidden_dim)
        
        # Apply softmax so the relationships for each patient sum to 1.0
        A = torch.softmax(scores, dim=-1) # A shape: [N, N]
        
        # Randomly drop relationships during training 
        A = self.attn_drop(A)
        
        return A

class ModulationBlock(nn.Module):
    def __init__(self, embed_dim, cnn_channels):
        super().__init__()
        
        self.channel_gate = nn.Sequential(
            nn.Linear(embed_dim, cnn_channels),
            nn.Sigmoid()
        )

    def forward(self, cnn_features, blended_patches):
        # cnn_features shape: [N, cnn_channels, Depth, Height, Width]
        # blended_patches shape: [N, num_patches, embed_dim]
        
        # Average across the patch dimension to get a global graph summary
        global_summary = blended_patches.mean(dim=1) # Shape becomes: [N, embed_dim]
        
        # Calculate the volume knobs for each channel
        scale = self.channel_gate(global_summary)
        
        # Reshape the scale so PyTorch can broadcast it across the 3D dimensions
        scale = scale.view(scale.size(0), -1, 1, 1, 1)
        
        # Modulate the original CNN features and add the residual connection
        modulated_features = (cnn_features * scale) + cnn_features
        
        return modulated_features
    
class HyperFuseNet(nn.Module):
    def __init__(self, num_layers=3, num_classes=6, num_features=4, hidden_dim=256, base_channels=16):
        super().__init__()
        # Internal hyperparameters
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.base_channels = base_channels
        self.num_features = num_features

        self.cnn_layers = nn.ModuleList()
        self.internal_attention = nn.ModuleList()
        self.modulation_blocks = nn.ModuleList()
        
        # Establish the starting size. A raw MRI always has 1 channel.
        current_in = 1 
        
        for i in range(self.num_layers):
            # Calculate how large the output will be at this specific depth
            current_out = self.base_channels * (2 ** (i+1))
            
            # Build the blocks so they match the exact data sizes
            self.cnn_layers.append(
                CNN_Block(in_channels=current_in, out_channels=current_out)
            )

            # The modulation block needs to know the exact token size (current_out) to map it back to the exact number of CNN channels (current_out)
            self.modulation_blocks.append(
                ModulationBlock(embed_dim=current_out, cnn_channels=current_out)
            )
            
            # Update the tracking variable so the next layer seamlessly connects
            current_in = current_out


        # The clinical attention block needs to know how many raw clinical features there are, and what hidden dimension to project them into
        self.clinical_attention = ClinicalEdgeAttention(clinical_features=self.num_features, hidden_dim=self.hidden_dim)

        # Dropout to prevent overfitting on the small dataset
        self.dropout = nn.Dropout(p=0.1)

        # A standard linear classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),            # Squish the 12x12x12 cube into 1x1x1
            nn.Flatten(),                       # Flatten to [Batch, 128]
            nn.Linear(current_out, hidden_dim), # current_out is 128
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images, clinical_data):
        # Compute the adjacency matrix A from the clinical data
        A = self.clinical_attention(clinical_data)
        current_features = images
        
        # We process the heavy 3D math in chunks to protect the GPU memory
        chunk_size = 32

        for i in range(self.num_layers):
            spatial_features_list = []
            raw_patches_list = []
            
            # Process heavy convolutions and internal attention in chunks
            for chunk in torch.split(current_features, chunk_size, dim=0):
                # Gradient checkpointing throws away intermediate memory and recomputes later
                sp_feat, r_patches = checkpoint(self.cnn_layers[i], chunk, use_reentrant=False)
                
                spatial_features_list.append(sp_feat)
                raw_patches_list.append(r_patches)

            # Glue the chunks back together into the full patient population
            spatial_features = torch.cat(spatial_features_list, dim=0)
            raw_patches = torch.cat(raw_patches_list, dim=0)

            # Interbrain patchwise attention (Requires all brains at once)
            blended_patches = torch.einsum('ij, jpe -> ipe', A, raw_patches)

            # Modulate the CNN features
            current_features = self.modulation_blocks[i](spatial_features, blended_patches)

            # Apply dropout to the features before the next layer to prevent overfitting
            current_features = self.dropout(current_features)

        # pass the features straight through the linear head to get standard logits
        logits = self.classifier(current_features)
        
        return logits