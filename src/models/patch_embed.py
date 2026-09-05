import math
import torch
import torch.nn as nn


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int or tuple of (D, H, W)
    embed_dim: output dimension for each position
    returns: (D*H*W, embed_dim) or (1 + D*H*W, embed_dim)
    """
    if isinstance(grid_size, int):
        grid_d = grid_h = grid_w = grid_size
    else:
        grid_d, grid_h, grid_w = grid_size

    # Ensure each axis gets an even number of dimensions and they sum to embed_dim
    d_dim = 2 * (embed_dim // 6)
    h_dim = 2 * (embed_dim // 6)
    w_dim = embed_dim - d_dim - h_dim
    assert w_dim % 2 == 0, f"embed_dim {embed_dim} must allow even division across axes"

    def get_1d_sincos_pos_embed_from_grid(dim, pos):
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega /= (dim / 2.0)
        omega = 1.0 / (10000 ** omega)  # (dim/2,)
        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum("m,d->md", pos, omega)  # (M, dim/2)
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)  # (M, dim)

    # 3D grid
    grid_d_coords = torch.arange(grid_d, dtype=torch.float32)
    grid_h_coords = torch.arange(grid_h, dtype=torch.float32)
    grid_w_coords = torch.arange(grid_w, dtype=torch.float32)

    grid = torch.meshgrid(grid_d_coords, grid_h_coords, grid_w_coords, indexing="ij")
    d_pos = grid[0].reshape(-1)
    h_pos = grid[1].reshape(-1)
    w_pos = grid[2].reshape(-1)

    emb_d = get_1d_sincos_pos_embed_from_grid(d_dim, d_pos)
    emb_h = get_1d_sincos_pos_embed_from_grid(h_dim, h_pos)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_dim, w_pos)

    pos_embed = torch.cat([emb_d, emb_h, emb_w], dim=1)  # (D*H*W, embed_dim)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


class PatchEmbed3D(nn.Module):
    """
    3D Volumetric Patch Embedding.
    Projects volume (B, in_chans, D, H, W) -> (B, num_patches, embed_dim).
    """
    def __init__(self, img_size=(16, 16, 16), patch_size=(4, 4, 4), in_chans=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_volume = in_chans * patch_size[0] * patch_size[1] * patch_size[2]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 3D Convolution with kernel_size = patch_size and stride = patch_size
        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        assert (D, H, W) == self.img_size, f"Input volume dimensions {(D, H, W)} != expected {self.img_size}"
        
        # Conv3d -> (B, embed_dim, grid_d, grid_h, grid_w)
        x = self.proj(x)
        # Flatten spatial dimensions -> (B, embed_dim, num_patches) -> transpose to (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class RandomMasking3D(nn.Module):
    """
    Random patch masking module for 3D MAE.
    Default masking ratio: 75% (0.75).
    """
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        """
        x: (B, N, D)
        returns:
            x_visible: (B, N_keep, D)
            mask: (B, N) binary mask (0 = kept/visible, 1 = masked)
            ids_restore: (B, N) indices to recover original order
        """
        N, L, D = x.shape
        len_keep = int(round(L * (1 - self.mask_ratio)))
        # Guard against zero kept tokens
        len_keep = max(1, min(L, len_keep))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore, ids_keep
