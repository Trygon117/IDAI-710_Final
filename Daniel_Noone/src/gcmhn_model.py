# Gated Cross-Modal Hypergraph Network (GCMHN)

import torch
import torch.nn as nn
import torch.nn.functional as F



# MRI Encoding
# 3D tens --> vector rep

class mri_enc3d(nn.Module):
    def __init__(self, emb_dim = 128):
        super().__init__()

        self.features = nn.Sequential(
            # 3d conv block 1
            nn.Conv3d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size = 2), # og 96 to 48 dims

            # 3d conv block 2
            nn.Conv3d(16,32,3,padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2), #  48 to 24

            # 3d cinv blk 3
            nn.Conv3d(32,64,3,padding = 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # coll spatial dims 
            nn.AdaptiveAvgPool3d(1),

            # flat --> shape to batch, 64
            nn.Flatten()
        )

        self.projection = nn.Linear(64, emb_dim)

    def forward(self, img):
        x = self.features(img)
        z_i = self.projection(x)
        return z_i


# CLINICAL HYPEREDGE BUILDER
# tab h-edge build
# using tabular data --> propose soft h-edge members
# H[i,m] --> how strong px i belongs to hyperegde/clin subgroup m
# tabular data suggesting relationships

class clin_hyperedge_build(nn.Module):
    def __init__(self, clin_dim, n_hedge = 16, hidden = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(clin_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_hedge))

    def forward(self, tab_data):
        logits = self.net(tab_data) # getting soft/raw socres for hyper edge members

        # using softma so q Px memberships sum = 1
        H_clin = F.softmax(logits, dim=1)

        return H_clin

# MRI GATING
# using mri embeds to refine/correct clinical h-egdes
# gate says if mri evidence supports or weakens Px i's member in subgroup/hedge m
# clinical suggests H. MRI data gates/refines H

class mri_gate(nn.Module):
    def __init__(self, emb_dim = 128, n_hedge = 16, hidden = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_hedge),

            nn.Sigmoid() # will giv val betwn 0,1 (suppres, support/keep) rels
        )

    def forward(self, z_i):
        gate = self.net(z_i)
        return gate


# SOFT Hypergraph Convl
# perf message passing thru learned hyperedges

class soft_hgraph_conv(nn.Module):
    def __init__(self, dim_input, dim_output, n_hedge):
        super().__init__()

        # trans Px embeds after message passing
        self.linear = nn.Linear(dim_input, dim_output)

        # importance score to learn for each h-edge
        self.edge_wts = nn.Parameter(torch.ones(n_hedge))

    def forward(self, Z, H): # Z is Px mri embeds, H is final gated h-edge memberships
        eps = 1e-8

        B, M = H.shape # getting batch size and num hyper edges

        w_edge = F.softplus(self.edge_wts) + eps # making wts pos

        # applyig wts to H
        HW = H * w_edge.unsqueeze(0) # scalig each col of H by learned wt (importance)

        # calc node degree, --> how much total h-edge membership each Px has
        Dn = HW.sum(dim=1).clamp(min=eps)

        # calc h-edge degree, --> how much each Px depends to q h-edge
        De = H.sum(dim=0).clamp(min=eps)

        # std h-edges 
        H_std = HW / De.unsqueeze(0)

        # calc eff Px-Px prop mx
        A_h = H_std @ H.T # --> [i,j] lg if Px i & Px j share similar memberships

        # std by node deg
        A_h = A_h / Dn.unsqueeze(1)

        # MESSAGE PASSING
        Z_mp = A_h @ Z # each Px emb --> wtd mixture of other Pxs embs (based on shared h-edges)

        # Trans features w learnable NN wts
        Z_op = self.linear(Z_mp)

        # adding nonlin w relu
        Z_op = F.relu(Z_op)

        return Z_op, A_h


# Gated Cross-Modal Hypergraph Network (GCMHN)
# 1) MRI encoded --> Z
# 2) Clin Tab data used to build h-edges --> H_clin
# 3) MRI gating --> gate
# 4) final h-edges --> H = std(H_clin * gate)
# 5) hypergraph convl --> Z
# 6) CLF --> Z_fin --> logits  

class GCMHN(nn.Module):
    def __init__(self, clin_dim, n_classes = 6, emb_dim = 128, n_hedge = 16, hidden = 64, dropout = 0.3):
        super().__init__()

        # saving for model reload
        self.clin_dim = clin_dim
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.n_hedge = n_hedge
        self.hidden = hidden
        self.dropout = dropout
    
        # MRI to vec emb
        self.mri_encoder = mri_enc3d(emb_dim = emb_dim)
    
        # clin data to soft h-edge propositions
        self.hedge_builder = clin_hyperedge_build(clin_dim = clin_dim, n_hedge = n_hedge, hidden = hidden)
    
        # mri gating --> refine/correct h-edge memberships of Pxs
        self.mri_gater = mri_gate(emb_dim = emb_dim, n_hedge = n_hedge, hidden = hidden)
    
        # hgraph mp layer
        self.hgl = soft_hgraph_conv(dim_input = emb_dim, dim_output = emb_dim, n_hedge = n_hedge)
    
        # final clfr
        self.clf = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, img, tab_data): # img: (B,1,96,96,96), tab/clin: (B, clin_dim)
        # MRI ENCODE
        Z = self.mri_encoder(img)

        # Clin data h-edges
        H_clin = self.hedge_builder(tab_data)

        # mri generated gates
        G = self.mri_gater(Z)

        # using gatees to refine/correct clin h-edges
        H = H_clin * G

        # re-std to make q Px h-edges sum = 1
        H = H / (H.sum(dim=1, keepdim=True) + 1e-8)

        # hgraph mp
        Z, A_h = self.hgl(Z,H)

        # clfr --> logits
        logits = self.clf(Z)

        return logits
        

    def save_model(self, path, notes = None):
        torch.save(self.state_dict(), path)


