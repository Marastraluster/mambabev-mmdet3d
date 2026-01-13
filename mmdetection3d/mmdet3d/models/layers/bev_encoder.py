from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SimpleBEVEncoder(nn.Module):
    """
    A minimal, runnable BEV encoder for MambaBEV (placeholder version).

    This encoder:
        - Takes multi-camera image features
        - Aggregates them by simple averaging
        - Projects them to a fixed BEV grid via convolution

    Note:
        - img_metas and prev_bev are currently unused
        - No geometric projection or temporal alignment is performed
        - This module is intended as a placeholder to keep the pipeline runnable
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        bev_h: int = 50,
        bev_w: int = 50,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bev_h = bev_h
        self.bev_w = bev_w

        # simple conv projection to BEV feature space
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        img_feats: List[torch.Tensor],
        img_metas: Optional[List[dict]] = None,
        prev_bev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            img_feats:
                list of multi-scale image features.
                The last level is used:
                shape = (B, N_cam, C, H, W)

            img_metas:
                unused (kept for interface compatibility)

            prev_bev:
                unused (kept for interface compatibility)

        Returns:
            bev_feat: (B, out_channels, bev_h, bev_w)
        """
        # use the last (highest-level) feature
        feat = img_feats[-1]  # (B, N_cam, C, H, W)
        B, N, C, H, W = feat.shape

        # defensive check: channel consistency
        assert (
            C == self.in_channels
        ), f"Input channel mismatch: expected {self.in_channels}, got {C}"

        # aggregate multi-camera features (simple average)
        feat = feat.mean(dim=1)  # (B, C, H, W)

        # keep dtype consistent with conv weights (AMP safety)
        feat = feat.to(dtype=self.proj[0].weight.dtype)

        # resize to BEV grid size
        feat = F.interpolate(
            feat,
            size=(self.bev_h, self.bev_w),
            mode="bilinear",
            align_corners=False,
        )

        # project to BEV feature
        bev_feat = self.proj(feat)  # (B, out_channels, bev_h, bev_w)

        return bev_feat
