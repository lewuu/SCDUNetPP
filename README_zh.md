# SCDUNetPP 中文使用指南

成都理工大学滑坡遥感语义分割模型。SCDUNetPP 是融合 CNN 与 Swin Transformer 的混合架构，用于滑坡测绘（滑坡边界提取）。

论文: JAG 2024 | 仓库: github.com/lewuu/SCDUNetPP

## 环境配置

```bash
git clone https://github.com/lewuu/SCDUNetPP.git
cd SCDUNetPP
pip install -r requirements.txt
```

核心依赖: torch >= 1.10, numpy, opencv-python, tqdm

## 数据准备

原始数据为泸定地区滑坡标注（多通道遥感 + DEM 地形数据）。

```
data/
└── luding/
    ├── image/    # 多通道遥感影像 (.tif)
    ├── mask/     # 滑坡标注 (.png, 0=背景 1=滑坡)
    └── dem/      # DEM 地形数据 (.tif)
```

### 自定义数据集

替换 `configs/config_case_luding.py` 中的数据路径和通道配置：

```python
# configs/config_case_luding.py
cfg.data_root = "data/your_region/"
cfg.in_channels = 6  # 3 波段 + 3 地形特征
```

## 训练

```bash
python train.py --config config_case_luding.py
```

关键参数（在 config 文件中调整）：
- `cfg.optimizer.base_lr`: 初始学习率（默认 1e-4）
- `cfg.train.epochs`: 训练轮数
- `cfg.train.batch_size`: 批大小（显存不足时减小）
- `cfg.model.name`: 模型选择（SCDUNetPP / UNet / DeepLabV3+）

## 测试

```bash
python test.py --config config_case_luding.py --checkpoint checkpoints/best.pth
```

输出: `results/` 目录下的预测掩码和评估指标。

## 模型架构

SCDUNetPP 的核心模块：

- **GLFE (Global-Local Feature Extractor)**: Swin Transformer 提取全局上下文
- **DSSA (Dual-Scale Spatial Attention)**: 空间注意力增强边界
- **DSC (Deep Supervision Contour)**: 深监督 + 轮廓约束

## 常见问题

### 显存不足
减小 batch_size 或输入图像尺寸 (`cfg.data.crop_size`)

### 数据集格式不匹配
确保 mask 值为 0/1（二值图），影像为多通道 .tif

### 预训练权重
联系作者获取或使用默认随机初始化
