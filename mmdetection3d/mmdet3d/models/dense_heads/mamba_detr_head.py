from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


def _build_mamba_block(d_model: int):
    """Build Mamba block if mamba-ssm is available; otherwise fallback to Identity."""
    try:
        from mamba_ssm import Mamba
        return Mamba(d_model=d_model)
    except Exception:
        return nn.Identity()


def _sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Minimal focal loss for multi-label classification."""
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


@MODELS.register_module()
class MambaDETRHead(nn.Module):
    """
    Minimal runnable DETR-like head for MambaBEV (v1.1.0 compatible).

    Note:
        - Matching strategy is a placeholder (naive first-K matching)
        - Intended to keep the pipeline runnable, not for final performance
    """

    def __init__(
        self,
        num_classes: int = 10,
        embed_dims: int = 256,
        num_queries: int = 900,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        ffn_dims: int = 1024,
        bbox_dim: int = 9,
        mamba_on_queries: bool = True,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 5.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.bbox_dim = bbox_dim

        self.query_embed = nn.Embedding(num_queries, embed_dims)

        self.mamba = _build_mamba_block(embed_dims) if mamba_on_queries else nn.Identity()

        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)
             for _ in range(num_decoder_layers)]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dims, ffn_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(ffn_dims, embed_dims),
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_decoder_layers)])

        self.cls_out = nn.Linear(embed_dims, num_classes)
        self.bbox_out = nn.Linear(embed_dims, bbox_dim)

        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight

    def _bev_to_tokens(self, bev_feat: torch.Tensor) -> torch.Tensor:
        return bev_feat.flatten(2).transpose(1, 2).contiguous()

    def _decode(self, bev_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = bev_feat.size(0)
        memory = self._bev_to_tokens(bev_feat)

        q = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        q = self.mamba(q).contiguous()

        for i in range(self.num_decoder_layers):
            attn_out, _ = self.cross_attn[i](q, memory, memory, need_weights=False)
            q = self.norm[i](q + attn_out + self.ffn[i](q))

        return self.cls_out(q), self.bbox_out(q)

    def forward_train(
        self,
        bev_feat: torch.Tensor,
        img_metas: List[Dict],
        gt_bboxes_3d: Optional[List] = None,
        gt_labels_3d: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        cls_logits, bbox_pred = self._decode(bev_feat)

        device = cls_logits.device
        B, Q, _ = cls_logits.shape

        cls_target = torch.zeros((B, Q, self.num_classes), device=device)
        bbox_target = torch.zeros((B, Q, self.bbox_dim), device=device)
        bbox_mask = torch.zeros((B, Q), device=device, dtype=torch.bool)

        if gt_bboxes_3d is not None and gt_labels_3d is not None:
            for b in range(B):
                labels = gt_labels_3d[b].to(device)
                bboxes_tensor = gt_bboxes_3d[b].tensor.to(device)
                num_gt = bboxes_tensor.size(0)
                k = min(num_gt, Q)
                if k > 0:
                    cls_target[b, torch.arange(k, device=device), labels[:k]] = 1.0
                    bbox_target[b, :k] = bboxes_tensor[:k, : self.bbox_dim]
                    bbox_mask[b, :k] = True

        loss_cls = _sigmoid_focal_loss(cls_logits, cls_target) * self.loss_cls_weight

        if bbox_mask.any():
            loss_bbox = F.l1_loss(
                bbox_pred[bbox_mask], bbox_target[bbox_mask]
            ) * self.loss_bbox_weight
        else:
            loss_bbox = bbox_pred.sum() * 0.0

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    @torch.no_grad()
    def forward_test(
        self,
        bev_feat: torch.Tensor,
        img_metas: List[Dict],
        score_thr: float = 0.05,
        topk: int = 300,
        **kwargs,
    ):
        cls_logits, bbox_pred = self._decode(bev_feat)
        B, Q, C = cls_logits.shape
        prob = torch.sigmoid(cls_logits)

        results = []
        for b in range(B):
            p = prob[b].reshape(-1)
            scores, inds = torch.topk(p, k=min(topk, p.numel()))
            keep = scores > score_thr
            scores, inds = scores[keep], inds[keep]

            labels = inds % self.num_classes
            qinds = inds // self.num_classes
            bboxes = bbox_pred[b, qinds]

            results.append(
                dict(
                    boxes_3d=bboxes,
                    scores_3d=scores,
                    labels_3d=labels,
                )
            )
        return results
