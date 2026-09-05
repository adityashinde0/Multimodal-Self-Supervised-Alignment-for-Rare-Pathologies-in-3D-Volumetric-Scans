import torch
import torch.nn as nn
import torch.nn.functional as F


class Supervised3DBaseline(nn.Module):
    """
    Supervised 3D Convolutional Baseline for rare pathology classification
    and metric comparison as required by PRD.md and ARCHITECTURE.md.
    """
    def __init__(self, in_chans=1, num_classes=5, embed_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            # (B, 1, 16, 16, 16) -> (B, 32, 16, 16, 16)
            nn.Conv3d(in_chans, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # (B, 32, 16, 16, 16) -> (B, 32, 8, 8, 8)
            nn.MaxPool3d(kernel_size=2, stride=2),

            # (B, 32, 8, 8, 8) -> (B, 64, 8, 8, 8)
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # (B, 64, 8, 8, 8) -> (B, 64, 4, 4, 4)
            nn.MaxPool3d(kernel_size=2, stride=2),

            # (B, 64, 4, 4, 4) -> (B, 128, 4, 4, 4)
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # Global Average Pooling -> (B, 128, 1, 1, 1)
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.embed_dim = embed_dim
        self.proj = nn.Linear(128, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def extract_features(self, x):
        # x: (B, 1, D, H, W)
        feat = self.features(x)
        feat = feat.flatten(1)
        emb = self.proj(feat)
        norm_emb = F.normalize(emb, p=2, dim=-1)
        return norm_emb

    def forward(self, x):
        norm_emb = self.extract_features(x)
        logits = self.classifier(norm_emb)
        return logits, norm_emb
