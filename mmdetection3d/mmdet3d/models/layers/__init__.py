# Copyright (c) OpenMMLab. All rights reserved.
from .box3d_nms import (aligned_3d_nms, box3d_multiclass_nms, circle_nms,
                        nms_bev, nms_normal_bev)
from .dgcnn_modules import DGCNNFAModule, DGCNNFPModule, DGCNNGFModule
from .edge_fusion_module import EdgeFusionModule
from .mlp import MLP
from .norm import NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d
from .paconv import PAConv, PAConvCUDA
from .pointnet_modules import (PAConvCUDASAModule, PAConvCUDASAModuleMSG,
                               PAConvSAModule, PAConvSAModuleMSG,
                               PointFPModule, PointSAModule, PointSAModuleMSG,
                               build_sa_module)
from .sparse_block import (SparseBasicBlock, SparseBottleneck,
                           make_sparse_convmodule)
from .torchsparse_block import TorchSparseConvModule, TorchSparseResidualBlock
from .transformer import GroupFree3DMHA
from .vote_module import VoteModule

# =========================
# 🔽 新增：MambaBEV layers
# =========================
from .temporal_mamba import TemporalMamba
from .bev_encoder import SimpleBEVEncoder
from .bev_positional_encoding import BEVPositionalEncoding

__all__ = [
    'VoteModule', 'GroupFree3DMHA', 'EdgeFusionModule', 'DGCNNFAModule',
    'DGCNNFPModule', 'DGCNNGFModule', 'NaiveSyncBatchNorm1d',
    'NaiveSyncBatchNorm2d', 'PAConv', 'PAConvCUDA', 'SparseBasicBlock',
    'SparseBottleneck', 'make_sparse_convmodule', 'MLP',
    'box3d_multiclass_nms', 'aligned_3d_nms', 'circle_nms', 'nms_bev',
    'nms_normal_bev', 'build_sa_module', 'PointSAModuleMSG',
    'PointSAModule', 'PointFPModule', 'PAConvSAModule', 'PAConvSAModuleMSG',
    'PAConvCUDASAModule', 'PAConvCUDASAModuleMSG',
    'TorchSparseConvModule', 'TorchSparseResidualBlock',
    # 🔽 新增
    'TemporalMamba',
    'SimpleBEVEncoder',
    'BEVPositionalEncoding',
]
