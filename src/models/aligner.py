import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mae3d import MaskedAutoencoder3D
from .text_encoder import ClinicalReportEncoder
from .projector import ProjectionHead


class Multimodal3DAligner(nn.Module):
    """
    Multimodal Self-Supervised 3D Vision-Language Aligner.
    Combines:
      - 3D Masked Autoencoder (MAE) with ~75% volumetric masking
      - Clinical Text Encoder
      - Shared metric projection heads
      - Symmetric InfoNCE contrastive alignment
    """
    def __init__(
        self,
        img_size=(16, 16, 16),
        patch_size=(4, 4, 4),
        embed_dim=128,
        shared_dim=128,
        text_model_name=None,
        mask_ratio=0.75,
        recon_weight=1.0,
        initial_temp=0.07
    ):
        super().__init__()
        self.recon_weight = recon_weight
        
        # 1. Visual Branch (3D-MAE)
        self.visual_encoder = MaskedAutoencoder3D(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            mask_ratio=mask_ratio
        )
        self.visual_projector = ProjectionHead(
            in_dim=embed_dim,
            hidden_dim=embed_dim * 2,
            out_dim=shared_dim
        )

        # 2. Textual Branch
        self.text_encoder = ClinicalReportEncoder(
            model_name=text_model_name,
            embed_dim=embed_dim
        )
        self.text_projector = ProjectionHead(
            in_dim=embed_dim,
            hidden_dim=embed_dim * 2,
            out_dim=shared_dim
        )

        # 3. Learnable logit scale / temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / initial_temp))

    def get_image_embedding(self, volumes):
        """
        Extract normalized 3D visual embedding for retrieval.
        volumes: (B, 1, D, H, W)
        returns: (B, shared_dim)
        """
        # Encode unmasked full volume
        raw_feat = self.visual_encoder.encode_volume(volumes)
        norm_emb = self.visual_projector(raw_feat)
        return norm_emb

    def get_text_embedding(self, reports):
        """
        Extract normalized text embedding for query retrieval.
        reports: list of str or tensor of tokens
        returns: (B, shared_dim)
        """
        raw_feat = self.text_encoder(reports)
        norm_emb = self.text_projector(raw_feat)
        return norm_emb

    def forward_contrastive_loss(self, img_emb, text_emb):
        """
        Symmetric InfoNCE Loss.
        img_emb: (B, shared_dim), normalized
        text_emb: (B, shared_dim), normalized
        """
        B = img_emb.shape[0]
        logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)

        # Cosine similarity matrix: (B, B)
        logits_per_image = logit_scale * torch.matmul(img_emb, text_emb.T)
        logits_per_text = logits_per_image.T

        labels = torch.arange(B, device=img_emb.device, dtype=torch.long)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        return contrastive_loss, logits_per_image

    def forward(self, volumes, reports, mask_ratio=None):
        """
        Joint forward pass for multimodal self-supervised training.
        """
        # 1. 3D-MAE forward pass with masked reconstruction
        mae_out = self.visual_encoder(volumes, mask_ratio=mask_ratio)
        recon_loss = mae_out["loss"]
        
        # Visual embedding from full volume representation
        vis_raw = self.visual_encoder.encode_volume(volumes)
        img_emb = self.visual_projector(vis_raw)

        # 2. Text branch forward pass
        text_raw = self.text_encoder(reports)
        text_emb = self.text_projector(text_raw)

        # 3. Symmetric Contrastive Alignment
        contrastive_loss, sim_matrix = self.forward_contrastive_loss(img_emb, text_emb)

        # Total combined loss
        total_loss = contrastive_loss + self.recon_weight * recon_loss

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "recon_loss": recon_loss,
            "image_embedding": img_emb,
            "text_embedding": text_emb,
            "sim_matrix": sim_matrix,
            "pred": mae_out["pred"],
            "mask": mae_out["mask"]
        }
