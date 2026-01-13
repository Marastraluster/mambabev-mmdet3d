from typing import Dict, List, Optional

import torch

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors import Base3DDetector


@MODELS.register_module()
class MambaBEV(Base3DDetector):
    """
    MambaBEV detector (mmdetection3d v1.1.0 compatible)

    Responsibilities:
        - Orchestrate backbone / neck / BEV encoder
        - Call TemporalMamba (models/layers)
        - Call detection head
    """

    def __init__(
        self,
        backbone: Dict,
        neck: Optional[Dict] = None,
        bev_encoder: Optional[Dict] = None,
        temporal_fusion: Optional[Dict] = None,
        bbox_head: Optional[Dict] = None,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        # ---------------- build modules via registry ----------------
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        self.bev_encoder = MODELS.build(bev_encoder) if bev_encoder is not None else None
        self.temporal_fusion = (
            MODELS.build(temporal_fusion) if temporal_fusion is not None else None
        )
        self.bbox_head = MODELS.build(bbox_head) if bbox_head is not None else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # ---------------- internal state ----------------
        # cache previous BEV for temporal fusion
        self.prev_bev = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_img_feat(self, img: torch.Tensor):
        """
        Args:
            img: (B, N_cam, 3, H, W)

        Returns:
            img_feats: list of multi-scale features
                       each: (B, N_cam, C, H_i, W_i)
        """
        B, N, C, H, W = img.shape
        img = img.view(B * N, C, H, W)

        feats = self.backbone(img)

        if self.neck is not None:
            feats = self.neck(feats)

        img_feats = []
        for feat in feats:
            _, C_f, H_f, W_f = feat.shape
            feat = feat.view(B, N, C_f, H_f, W_f)
            img_feats.append(feat)

        return img_feats

    # ------------------------------------------------------------------
    # BEV construction
    # ------------------------------------------------------------------
    def extract_bev_feat(
        self,
        img_feats,
        img_metas: List[Dict],
    ):
        """
        Returns:
            bev_feat: (B, C, H_bev, W_bev)
        """
        if self.bev_encoder is None:
            raise RuntimeError("bev_encoder must be provided for MambaBEV")

        bev_feat = self.bev_encoder(
            img_feats=img_feats,
            img_metas=img_metas,
            prev_bev=self.prev_bev,
        )
        return bev_feat

    # ------------------------------------------------------------------
    # Forward train
    # ------------------------------------------------------------------
    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        **kwargs,
    ):
        """
        Training forward.
        """
        # reset temporal cache at first frame of a scene (if provided)
        if img_metas[0].get("is_first_frame", False) or \
           img_metas[0].get("scene_start", False):
            self.prev_bev = None

        # 1. image -> multi-view features
        img_feats = self.extract_img_feat(img)

        # 2. BEV encoding
        bev_feat = self.extract_bev_feat(img_feats, img_metas)

        # 3. temporal fusion (TemporalMamba)
        if self.temporal_fusion is not None:
            bev_feat = self.temporal_fusion(
                cur_bev=bev_feat,
                prev_bev=self.prev_bev,
                img_metas=img_metas,
            )

        # update BEV cache (detach to avoid cross-step backprop)
        self.prev_bev = bev_feat.detach()

        # 4. detection head
        losses = self.bbox_head.forward_train(
            bev_feat=bev_feat,
            img_metas=img_metas,
            **kwargs,
        )

        return losses

    # ------------------------------------------------------------------
    # Forward test
    # ------------------------------------------------------------------
    def forward_test(
        self,
        img: torch.Tensor,
        img_metas: List[Dict],
        **kwargs,
    ):
        """
        Inference forward.
        """
        # reset temporal cache at scene start or first frame
        if img_metas[0].get("is_first_frame", False) or \
           img_metas[0].get("scene_start", False):
            self.prev_bev = None

        img_feats = self.extract_img_feat(img)
        bev_feat = self.extract_bev_feat(img_feats, img_metas)

        if self.temporal_fusion is not None:
            bev_feat = self.temporal_fusion(
                cur_bev=bev_feat,
                prev_bev=self.prev_bev,
                img_metas=img_metas,
            )

        self.prev_bev = bev_feat.detach()

        results = self.bbox_head.forward_test(
            bev_feat=bev_feat,
            img_metas=img_metas,
            **kwargs,
        )

        # v1.1.0 expects List[dict]
        if not isinstance(results, list):
            results = [results]

        return results
