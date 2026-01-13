from typing import List, Optional

import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


# ---------------------------------------------------------
# Try to build a Mamba block (compatible with mamba-ssm 1.2.0)
# ---------------------------------------------------------
def build_mamba_block(d_model: int):
    try:
        from mamba_ssm import Mamba
        return Mamba(d_model=d_model)
    except Exception:
        # fallback: keep model runnable
        return nn.Identity()


# ---------------------------------------------------------
# BEV index utilities (four-direction rearrangement)
# ---------------------------------------------------------
def build_bev_permutations(h: int, w: int, device):
    # row-major
    idx_row = torch.arange(h * w, device=device)
    # col-major
    idx_col = torch.arange(h * w, device=device).view(h, w).t().reshape(-1)

    perms = {
        "fl": idx_row,                       # forward-left
        "fu": idx_col,                       # forward-up
        "rl": torch.flip(idx_row, dims=[0]), # reverse-left
        "ru": torch.flip(idx_col, dims=[0]), # reverse-up
    }
    return perms


def inverse_permutation(p: torch.Tensor):
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), device=p.device)
    return inv


# ---------------------------------------------------------
# TemporalMamba
# ---------------------------------------------------------
@MODELS.register_module()
class TemporalMamba(nn.Module):
    """
    TemporalMamba (mmdet3d v1.1.0 compatible)

    Note:
        - img_metas is currently unused (no ego-motion compensation yet)
        - This module only performs temporal feature fusion in BEV space
    """

    def __init__(
        self,
        embed_dims: int = 256,
        dropout: float = 0.9,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.dropout = dropout

        # ---------------- Conv compression block ----------------
        mid = embed_dims // 2

        self.conv3 = nn.Sequential(
            nn.Conv2d(embed_dims * 2, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dims * 2, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(mid * 2, embed_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
        )

        # ---------------- Sequence projection ----------------
        self.seq_proj = nn.Linear(embed_dims, embed_dims)
        self.seq_norm = nn.LayerNorm(embed_dims)

        # ---------------- Mamba ----------------
        self.mamba = build_mamba_block(embed_dims)

        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.out_dropout = nn.Dropout(dropout)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(
        self,
        cur_bev: torch.Tensor,
        prev_bev: Optional[torch.Tensor] = None,
        img_metas: Optional[List[dict]] = None,
    ):
        """
        Args:
            cur_bev: (B, C, H, W)
            prev_bev: (B, C, H, W) or None
            img_metas: unused (kept for interface compatibility)

        Returns:
            fused_bev: (B, C, H, W)
        """
        if prev_bev is None:
            return cur_bev

        B, C, H, W = cur_bev.shape
        device = cur_bev.device

        # 1. conv-based fusion
        x = torch.cat([cur_bev, prev_bev], dim=1)  # (B, 2C, H, W)
        x3 = self.conv3(x)
        x1 = self.conv1(x)
        z = self.fuse_conv(torch.cat([x3, x1], dim=1))  # (B, C, H, W)

        # 2. flatten to sequence
        z = z.flatten(2).transpose(1, 2)  # (B, HW, C)
        z = self.seq_norm(self.seq_proj(z))

        # 3. build permutations once
        perms = build_bev_permutations(H, W, device)
        inv_perms = {k: inverse_permutation(v) for k, v in perms.items()}

        outs = []
        for key in ["fl", "fu", "rl", "ru"]:
            p = perms[key]
            inv_p = inv_perms[key]

            seq = z[:, p, :]
            seq = self.mamba(seq).contiguous()
            seq = self.out_proj(seq)
            seq = seq[:, inv_p, :]

            outs.append(seq)

        # 4. average four directions
        z = torch.stack(outs, dim=0).mean(dim=0)  # (B, HW, C)
        z = z.transpose(1, 2).reshape(B, C, H, W)

        # 5. residual + dropout
        z = self.out_dropout(z)
        out = cur_bev + z

        return out
