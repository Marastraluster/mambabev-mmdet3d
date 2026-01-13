# mmdetection3d/mmdet3d/models/layers/bev_positional_encoding.py

from typing import Optional

import math
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVPositionalEncoding(nn.Module):
    """
    BEV positional encoding module.

    Supports:
        - 'learnable': learnable BEV positional embedding
        - 'sine': fixed sine-cosine positional encoding

    Input:
        bev_feat: (B, C, H, W)

    Output:
        pos_embed: (1, C, H, W)  (broadcastable)
    """

    def __init__(
        self,
        embed_dims: int = 256,
        bev_h: int = 50,
        bev_w: int = 50,
        mode: str = "learnable",
    ):
        super().__init__()

        assert mode in ["learnable", "sine"], \
            f"Unsupported BEV positional encoding mode: {mode}"

        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.mode = mode

        if mode == "learnable":
            # (1, C, H, W)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, embed_dims, bev_h, bev_w)
            )
            nn.init.normal_(self.pos_embed, std=0.02)

        else:
            # sine-cosine has no learnable params
            self.register_buffer(
                "pos_embed",
                self._build_sine_positional_encoding(),
                persistent=False,
            )

    def _build_sine_positional_encoding(self) -> torch.Tensor:
        """
        Build fixed sine-cosine BEV positional encoding.

        Returns:
            pos_embed: (1, C, H, W)
        """
        h, w = self.bev_h, self.bev_w
        c = self.embed_dims
        assert c % 4 == 0, "embed_dims must be divisible by 4 for sine encoding"

        # grid
        y_embed = torch.arange(h).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w).unsqueeze(0).repeat(h, 1)

        # normalize
        eps = 1e-6
        y_embed = y_embed / (h + eps)
        x_embed = x_embed / (w + eps)

        dim_t = torch.arange(c // 4, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (c // 2))

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack(
            (pos_x.sin(), pos_x.cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y.sin(), pos_y.cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2)  # (H, W, C)
        pos = pos.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return pos

    def forward(self, bev_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            bev_feat: (B, C, H, W) or None (unused, kept for interface flexibility)

        Returns:
            pos_embed: (1, C, H, W)
        """
        return self.pos_embed
