import torch
import torch.nn as nn
from .patch_embed import PatchEmbed3D, RandomMasking3D, get_3d_sincos_pos_embed
from .attention import TransformerBlock


class MaskedAutoencoder3D(nn.Module):
    """
    3D Masked Autoencoder (3D-MAE) for Volumetric CT/MRI scans.
    Following He et al. (2021) adapted for 3D volumetric data.
    
    Default configuration:
      - Volume size: (16, 16, 16)
      - Patch size: (4, 4, 4) -> 64 tokens
      - Masking ratio: ~75% (48 masked, 16 visible tokens)
      - Encoder: 4 blocks, dim=128, 4 heads
      - Decoder: 2 blocks, dim=64, 4 heads
    """
    def __init__(
        self,
        img_size=(16, 16, 16),
        patch_size=(4, 4, 4),
        in_chans=1,
        embed_dim=128,
        depth=4,
        num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4.0,
        mask_ratio=0.75,
        norm_pix_loss=False
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # 1. Patch Embedding & Masking
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.patch_volume = self.patch_embed.patch_volume
        self.masker = RandomMasking3D(mask_ratio=mask_ratio)

        # 2. Encoder Positional Embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), 
            requires_grad=False
        )
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # 3. Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), 
            requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_volume, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize 3D Sin-Cos Positional Embeddings
        pos_embed = get_3d_sincos_pos_embed(self.embed_dim, self.patch_embed.grid_size)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_embed_dim, self.patch_embed.grid_size)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.unsqueeze(0))

        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize linear layers and LayerNorms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (B, 1, D, H, W)
        returns: (B, num_patches, patch_volume)
        """
        p_d, p_h, p_w = self.patch_size
        c = self.in_chans
        d = imgs.shape[2] // p_d
        h = imgs.shape[3] // p_h
        w = imgs.shape[4] // p_w

        # Reshape to (B, c, d, p_d, h, p_h, w, p_w)
        x = imgs.reshape(imgs.shape[0], c, d, p_d, h, p_h, w, p_w)
        # Permute to (B, d, h, w, c, p_d, p_h, p_w)
        x = torch.einsum("ncdphqwk->ndhwcpqk", x)
        # Flatten patches: (B, d*h*w, c*p_d*p_h*p_w)
        patches = x.reshape(imgs.shape[0], d * h * w, c * p_d * p_h * p_w)
        return patches

    def unpatchify(self, patches):
        """
        patches: (B, num_patches, patch_volume)
        returns: (B, 1, D, H, W)
        """
        p_d, p_h, p_w = self.patch_size
        c = self.in_chans
        d, h, w = self.patch_embed.grid_size
        assert d * h * w == patches.shape[1]

        x = patches.reshape(patches.shape[0], d, h, w, c, p_d, p_h, p_w)
        x = torch.einsum("ndhwcpqk->ncdphqwk", x)
        imgs = x.reshape(patches.shape[0], c, d * p_d, h * p_h, w * p_w)
        return imgs

    def forward_encoder(self, x, mask_ratio=None):
        """
        Encodes visible tokens only.
        """
        # Patch embed: (B, N, D)
        x = self.patch_embed(x)
        # Add position embedding
        x = x + self.pos_embed

        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        if mask_ratio > 0.0:
            # Masking: keep only visible tokens
            x_vis, mask, ids_restore, ids_keep = self.masker(x)
        else:
            x_vis = x
            mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
            ids_restore = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
            ids_keep = ids_restore

        # Apply Transformer blocks to visible tokens
        for blk in self.encoder_blocks:
            x_vis = blk(x_vis)
        x_vis = self.encoder_norm(x_vis)

        return x_vis, mask, ids_restore, ids_keep

    def forward_decoder(self, x_vis, ids_restore):
        """
        Decodes visible tokens + mask tokens into reconstructed patches.
        """
        # Embed tokens for decoder
        x = self.decoder_embed(x_vis)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        # Unshuffle to original order
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # Add decoder position embedding
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict patch voxels
        pred = self.decoder_pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: (B, 1, D, H, W)
        pred: (B, N, patch_volume)
        mask: (B, N), 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # mean loss per patch, (B, N)

        # Only compute loss on masked patches
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def encode_volume(self, imgs):
        """
        Inference / Representation extraction:
        Encodes full volume (no masking) and computes global average pooled visual embedding.
        returns: (B, embed_dim)
        """
        x = self.patch_embed(imgs) + self.pos_embed
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        # Global average pool across tokens: (B, embed_dim)
        visual_emb = x.mean(dim=1)
        return visual_emb

    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        latent, mask, ids_restore, ids_keep = self.forward_encoder(imgs, mask_ratio=mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        
        # Also compute pooled visual embedding from visible latents
        visual_emb = latent.mean(dim=1)

        return {
            "loss": loss,
            "pred": pred,
            "mask": mask,
            "latent": latent,
            "visual_embedding": visual_emb
        }
