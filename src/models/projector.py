import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    MLP projection head mapping modality representations to a shared metric space.
    Followed by L2-normalization for cosine similarity contrastive alignment.
    """
    def __init__(self, in_dim=128, hidden_dim=256, out_dim=128, drop=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )

    def forward(self, x):
        # x: (B, in_dim)
        feat = self.mlp(x)
        # L2-normalize to unit hypersphere
        norm_feat = F.normalize(feat, p=2, dim=-1)
        return norm_feat
