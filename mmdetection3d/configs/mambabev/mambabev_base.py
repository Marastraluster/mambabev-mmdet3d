_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py',
]

# ------------------------
# Global settings
# ------------------------
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_h = 100
bev_w = 100
embed_dims = 256
num_classes = 10
num_cams = 6

# ------------------------
# Data preprocessor (REQUIRED in v1.1.0)
# ------------------------
data_preprocessor = dict(
    type='Det3DDataPreprocessor',
    pad_size_divisor=32,
)

# ------------------------
# Model
# ------------------------
model = dict(
    type='MambaBEV',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=embed_dims,
        num_outs=1,
    ),
    bev_encoder=dict(
        type='SimpleBEVEncoder',
        in_channels=embed_dims,
        out_channels=embed_dims,
        bev_h=bev_h,
        bev_w=bev_w,
    ),
    temporal_fusion=dict(
        type='TemporalMamba',
        embed_dims=embed_dims,
        dropout=0.9,
    ),
    bbox_head=dict(
        type='MambaDETRHead',
        num_classes=num_classes,
        embed_dims=embed_dims,
        num_queries=900,
        num_decoder_layers=6,
        num_heads=8,
        ffn_dims=1024,
        bbox_dim=9,
        mamba_on_queries=True,
    ),
    train_cfg=dict(),
    test_cfg=dict(),
)

# ------------------------
# Optimizer
# ------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    ),
)

# ------------------------
# Training / Validation / Test loops
# ------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=1,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ------------------------
# Runtime
# ------------------------
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
)
