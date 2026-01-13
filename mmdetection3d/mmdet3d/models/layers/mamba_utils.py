# mmdetection3d/mmdet3d/models/layers/mamba_utils.py

"""
Utilities for Mamba-based modules in MambaBEV.

This file contains:
- Mamba block builder (compatible with mamba-ssm 1.2.0)
- BEV flatten / unflatten helpers
- Four-direction BEV sequence rearrangement utilities

NOTE:
- No nn.Module definitions here
- No registry usage here
- Pure utility functions only
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Mamba builder (safe for mamba-ssm 1.2.0)
# ------------------------------------------------------------------
def build_mamba_block(d_model: int) -> nn.Module:
    """
    Build a Mamba block if mamba-ssm is available.
    Fallback to Identity to keep model runnable.

    Args:
        d_model (int): embedding dimension

    Returns:
        nn.Module
    """
    try:
        from mamba_ssm import Mamba
        return Mamba(d_model=d_model)
    except Exception:
        return nn.Identity()


# ------------------------------------------------------------------
# BEV shape helpers
# ------------------------------------------------------------------
def bev_flatten(bev_feat: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """
    Flatten BEV feature map to sequence.

    Args:
        bev_feat: (B, C, H, W)

    Returns:
        seq: (B, H*W, C)
        H, W: spatial shape
    """
    B, C, H, W = bev_feat.shape
    seq = bev_feat.flatten(2).transpose(1, 2).contiguous()
    return seq, H, W


def bev_unflatten(seq: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Restore BEV feature map from sequence.

    Args:
        seq: (B, H*W, C)
        H, W: spatial shape

    Returns:
        bev_feat: (B, C, H, W)
    """
    B, L, C = seq.shape
    assert L == H * W, "Sequence length does not match H*W"
    bev_feat = seq.transpose(1, 2).reshape(B, C, H, W).contiguous()
    return bev_feat


# ------------------------------------------------------------------
# Four-direction BEV permutation utilities
# ------------------------------------------------------------------
def build_bev_permutations(
    H: int,
    W: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build index permutations for four-direction BEV scanning.

    Directions:
        - fl: forward-left   (row-major)
        - fu: forward-up     (column-major)
        - rl: reverse-left  (reverse row-major)
        - ru: reverse-up    (reverse column-major)

    Returns:
        dict[str, Tensor] with shape (H*W,)
    """
    # forward-left (row-major)
    idx_row = torch.arange(H * W, device=device)

    # forward-up (column-major)
    idx_col = torch.arange(H * W, device=device).view(H, W).t().reshape(-1)

    perms = {
        "fl": idx_row,
        "fu": idx_col,
        "rl": torch.flip(idx_row, dims=[0]),
        "ru": torch.flip(idx_col, dims=[0]),
    }
    return perms


def inverse_permutation(p: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse permutation.

    Args:
        p: permutation index tensor, shape (L,)

    Returns:
        inv_p: inverse permutation, shape (L,)
    """
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), device=p.device)
    return inv


# ------------------------------------------------------------------
# Four-direction rearrange / re-merge helpers
# ------------------------------------------------------------------
def bev_rearrange(
    seq: torch.Tensor,
    perm: torch.Tensor,
) -> torch.Tensor:
    """
    Rearrange BEV sequence.

    Args:
        seq: (B, L, C)
        perm: (L,)

    Returns:
        rearranged_seq: (B, L, C)
    """
    return seq[:, perm, :]


def bev_remerge(
    seq: torch.Tensor,
    inv_perm: torch.Tensor,
) -> torch.Tensor:
    """
    Restore BEV sequence to original order.

    Args:
        seq: (B, L, C)
        inv_perm: (L,)

    Returns:
        restored_seq: (B, L, C)
    """
    return seq[:, inv_perm, :]
