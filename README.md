# MambaBEV（mmdetection3d v1.1.0）论文复现 README

本仓库是在 **mmdetection3d v1.1.0 + PyTorch 2.0.0** 环境下，对 **MambaBEV** 论文进行的**工程级可运行复现**。

目标不是一次性对齐论文精度，而是：

> **先保证：代码结构正确、模块注册完整、config 合法、能够稳定训练与推理**，
> 再逐步替换占位模块，对齐论文实现。

---

## 1. 项目整体说明

### 1.1 核心特点

- 基于 **mmdetection3d v1.1.0（MMEngine）**
- 单卡 GPU（≈48GB 显存）可运行
- 不依赖自定义 CUDA / Triton / Flash-Attention
- 使用 **mamba-ssm 1.2.0**（可选，缺失时自动 fallback）
- 明确区分：
  - **占位实现（可运行）**
  - **论文对齐实现（后续替换）**

### 1.2 当前实现范围

| 模块 | 状态 | 说明 |
|----|----|----|
| Detector（MambaBEV） | ✅ | 工程级正确（v1.1.0） |
| Temporal Fusion（TemporalMamba） | ✅ | 四方向 BEV Mamba 扫描 |
| BEV Encoder | ✅ | 简化占位版（均值 + Conv） |
| Detection Head | ✅ | 简化 DETR-like Head |
| Config（tiny / base） | ✅ | 可直接 train |
| nuScenes 数据 | ⏳ | 需用户准备 |

---

## 2. 环境配置

### 2.1 基础环境

建议使用 **conda**：

```bash
conda create -n mambabev python=3.8 -y
conda activate mambabev
```

### 2.2 PyTorch

```bash
pip install torch==2.0.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ CUDA 版本需与服务器驱动匹配（示例为 CUDA 11.8）

### 2.3 mmdetection3d

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.1.0
pip install -v -e .
```

并安装依赖：

```bash
pip install mmengine==0.10.* mmcv>=2.0.0
pip install nuscenes-devkit
```

### 2.4 Mamba（可选）

```bash
pip install mamba-ssm==1.2.0
```

> 如果未安装或安装失败：
> - TemporalMamba / MambaDETRHead 会自动退化为 `nn.Identity`
> - **代码仍可运行（仅性能受影响）**

---

## 3. 代码结构说明

### 3.1 关键目录

```text
mmdet3d/
└── models/
    ├── detectors/
    │   └── mambabev.py
    ├── layers/
    │   ├── temporal_mamba.py
    │   ├── bev_encoder.py
    │   ├── bev_positional_encoding.py
    │   └── mamba_utils.py
    ├── dense_heads/
    │   └── mamba_detr_head.py
    └── __init__.py

configs/
└── mambabev/
    ├── mambabev_tiny.py
    └── mambabev_base.py
```

### 3.2 模块职责划分

- **MambaBEV（Detector）**  
  负责整体调度：backbone → BEV → temporal → head

- **SimpleBEVEncoder**  
  占位版 BEV 构建（后续可替换为论文 SCA / backward projection）

- **TemporalMamba**  
  BEV 时序融合（四方向 Mamba 扫描）

- **MambaDETRHead**  
  简化 DETR 风格 3D 检测头（可运行优先）

---

## 4. nuScenes 数据集准备

### 4.1 数据目录（必须）

```text
mmdetection3d/
└── data/
    └── nuscenes/
        ├── maps/
        ├── samples/
        ├── sweeps/
        ├── v1.0-trainval/
        ├── nuscenes_infos_train.pkl
        ├── nuscenes_infos_val.pkl
        └── nuscenes_infos_test.pkl
```

### 4.2 生成 infos.pkl

在 **mmdetection3d 根目录** 执行：

```bash
python tools/create_data.py nuscenes \
  --root-path data/nuscenes \
  --out-dir data/nuscenes \
  --extra-tag nuscenes
```

成功后会生成：

```text
nuscenes_infos_train.pkl
nuscenes_infos_val.pkl
```

---

## 5. 训练与测试

### 5.1 训练（推荐顺序）

#### ① 先跑 tiny（强烈推荐）

```bash
tools/train.py configs/mambabev/mambabev_tiny.py
```

特点：
- BEV: 50×50
- ResNet-50
- Decoder layers: 3
- 显存压力小，适合 debug

#### ② 再跑 base

```bash
tools/train.py configs/mambabev/mambabev_base.py
```

特点：
- BEV: 100×100
- ResNet-101
- Decoder layers: 6

---

### 5.2 常见“正常”报错说明

| 报错 | 含义 |
|----|----|
| FileNotFoundError: nuscenes | 数据未准备（正常） |
| KeyError: img | pipeline / infos.pkl 不匹配 |
| AssertionError: channel mismatch | neck 与 BEV encoder 配置不一致 |

---

## 6. 当前实现的已知限制

- ❌ 尚未实现论文中的 **SCA / backward projection**
- ❌ Temporal 未使用 ego-motion
- ❌ DETR Head 使用 naive matching（非 Hungarian）

这些均是 **刻意设计**，目的是：

> **先保证工程正确性，再逐步提升论文一致性**

---

## 7. 后续可扩展方向（建议顺序）

1. 替换 `SimpleBEVEncoder` → 论文版 SCA
2. TemporalMamba 中引入 ego-motion 对齐
3. MambaDETRHead 引入 Hungarian matching
4. 调整 BEV 分辨率 / queries 对齐论文

---

## 8. 总结

如果你能顺利运行：

```bash
tools/train.py configs/mambabev/mambabev_tiny.py
```

那么说明：

> **你的 MambaBEV 工程复现已经是“结构级成功”**，
> 后续工作将主要集中在性能与论文对齐，而不是工程 debug。

---

如需继续对齐论文实现（SCA / temporal / loss），可在此 README 基础上逐步扩展。

