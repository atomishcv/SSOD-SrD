# DINOv3 特征可视化模块说明

## 📁 文件结构

```
engine/solver/
├── visualize_features.py      # 🎨 特征可视化核心模块（独立文件）
├── supervised_loss.py          # 监督损失计算（调用可视化）
└── README_visualization.md     # 本说明文档
```

---

## 🎯 功能概述

`visualize_features.py` 提供了 DINOv3 backbone 多尺度特征图的热力图可视化功能。

### 核心函数

| 函数 | 功能 | 适用场景 |
|------|------|---------|
| `visualize_dinov3_features_simple()` | 简化版单样本可视化 | ✅ **推荐** 日常训练使用 |
| `visualize_dinov3_features()` | 完整版多样本可视化 | 详细分析（平均/最大/PCA） |
| `visualize_feature_on_image()` | 单特征图叠加 | 自定义可视化 |

---

## 🚀 快速使用

### 1. 在训练中自动可视化

已集成到 `supervised_loss.py`，训练时自动生成：

```python
# engine/solver/supervised_loss.py (已配置)
from .visualize_features import visualize_dinov3_features_simple

if 'backbone_feate' in outputs:
    visualize_dinov3_features_simple(
        backbone_feats=outputs['backbone_feate'],
        images=images,
        save_dir='./visualizations/dinov3_heatmaps',
        step=step,
        epoch=epoch,
        sample_idx=0,
        verbose=True
    )
```

**输出示例**：
```
./visualizations/dinov3_heatmaps/
├── epoch000_step00000_sample0.png
├── epoch000_step00100_sample0.png
└── ...
```

---

### 2. 单独调用可视化

```python
import torch
from engine.solver.visualize_features import visualize_dinov3_features_simple

# 假设你有特征和图像
backbone_feats = [
    torch.randn(8, 192, 64, 64),  # 高分辨率
    torch.randn(8, 192, 32, 32),  # 中分辨率
    torch.randn(8, 192, 16, 16),  # 低分辨率
]
images = torch.randn(8, 3, 640, 640)

# 调用可视化
visualize_dinov3_features_simple(
    backbone_feats=backbone_feats,
    images=images,
    save_dir='./my_visualizations',
    step=0,
    epoch=0,
    sample_idx=0,
    verbose=True  # 打印调试信息
)
```

---

### 3. 可视化单个特征图

```python
from engine.solver.visualize_features import visualize_feature_on_image

# 单个特征图 [C, H, W]
feature_map = torch.randn(192, 64, 64)
image = torch.randn(3, 640, 640)

result = visualize_feature_on_image(
    feature_map=feature_map,
    image=image,
    save_path='./my_heatmap.png',
    title='Custom Feature Heatmap',
    cmap='jet',
    alpha=0.6
)
```

---

## 🔧 参数说明

### `visualize_dinov3_features_simple()`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backbone_feats` | `List[Tensor]` | 必需 | 多尺度特征列表 `[B, C, H, W]` |
| `images` | `Tensor` | 必需 | 原始图像 `[B, 3, H, W]` |
| `save_dir` | `str` | `'./visualizations/dinov3_heatmaps'` | 保存目录 |
| `step` | `int` | `0` | 当前训练步数 |
| `epoch` | `int` | `0` | 当前 epoch |
| `sample_idx` | `int` | `0` | 要可视化的样本索引 |
| `verbose` | `bool` | `True` | 是否打印调试信息 |

---

## 📊 可视化效果

### 输出图像布局

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│             │   Scale 1   │   Scale 2   │   Scale 3   │
│   Original  │   64×64     │   32×32     │   16×16     │
│    Image    │  (↑10.0x)   │  (↑20.0x)   │  (↑40.0x)   │
│             │  blur=7     │  blur=11    │  blur=21    │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### 热力图颜色映射

- 🔴 **红色**：高激活区域（模型关注的重点）
- 🟡 **黄色**：中等激活
- 🔵 **蓝色**：低激活区域

---

## 🛠️ 技术细节

### 1. Float16 兼容性

自动处理混合精度训练产生的 `float16` 张量：

```python
# 自动转换
if feat_map.dtype == torch.float16:
    feat_map = feat_map.float()  # → float32

if avg_activation.dtype == np.float16:
    avg_activation = avg_activation.astype(np.float32)
```

### 2. 智能上采样策略

针对不同倍数的上采样采用不同策略：

```python
def _smart_resize(feature_map, target_size):
    upscale_ratio = target_h / feature_h
    
    if upscale_ratio > 4:
        # 多阶段上采样
        # Stage 1: INTER_LINEAR (2x)
        # Stage 2: INTER_CUBIC (4x)
        # Stage 3: GaussianBlur (平滑)
    else:
        # 单阶段 INTER_CUBIC
```

**优势**：
- ✅ 减少大倍数上采样的伪影
- ✅ 保持边缘平滑
- ✅ 突出关键特征

### 3. 错误处理

三层降级机制：

```python
try:
    # 优先：OpenCV (最快)
    heatmap = cv2.resize(...)
except:
    # 降级：scipy (更稳定)
    heatmap = zoom(...)
except:
    # 最终：保持原始尺寸
    heatmap = feature_map
```

---

## 🎨 自定义配置

### 修改保存目录

```python
# supervised_loss.py line 77
visualize_dinov3_features_simple(
    ...
    save_dir='./my_custom_dir',  # ← 修改这里
    ...
)
```

### 禁用详细输出

```python
visualize_dinov3_features_simple(
    ...
    verbose=False,  # ← 关闭调试信息
)
```

### 调整可视化频率

```python
# supervised_loss.py line 72
if 'backbone_feate' in outputs and step % 100 == 0:  # ← 每 100 步可视化一次
    visualize_dinov3_features_simple(...)
```

---

## 📝 调试信息示例

启用 `verbose=True` 时的输出：

```bash
🔍 [DINOv3可视化调试] Epoch 0, Step 100
   backbone_feats 类型: <class 'list'>
   backbone_feats 是否为 list: True
   backbone_feats 是否为 tuple: False
   ✅ backbone_feats 长度: 3
   [0] Tensor shape: [8, 192, 64, 64], device: cuda:0, dtype: torch.float32
   [1] Tensor shape: [8, 192, 32, 32], device: cuda:0, dtype: torch.float32
   [2] Tensor shape: [8, 192, 16, 16], device: cuda:0, dtype: torch.float32
   [0] avg_activation shape: (64, 64), dtype: float32
   [1] avg_activation shape: (32, 32), dtype: float32
   [2] avg_activation shape: (16, 16), dtype: float32
✅ 特征热力图已保存: ./visualizations/dinov3_heatmaps/epoch000_step00100_sample0.png
```

---

## ⚠️ 常见问题

### 1. 为什么热力图看起来很模糊？

**原因**：大倍数上采样（如 16×16 → 640×640 = 40倍）

**解决**：
- 这是正常现象，低分辨率特征本身就是粗粒度的
- 高斯平滑帮助减少块状伪影
- 关注高分辨率特征图（64×64）查看细节

### 2. 可视化失败：`cv2.error`

**原因**：OpenCV 不支持 `float16`

**解决**：已自动处理，如果仍然失败，检查：
```python
print(f"dtype: {feature_map.dtype}")  # 应该是 float32
```

### 3. 内存占用过高

**解决**：
- 降低可视化频率：`if step % 500 == 0`
- 减少保存的图像分辨率：`dpi=80`
- 仅可视化单个尺度

---

## 📚 相关文件

| 文件 | 说明 |
|------|------|
| `engine/solver/visualize_features.py` | 可视化核心实现 |
| `engine/solver/supervised_loss.py` | 集成调用点 |
| `engine/deim/deim.py` | DINOv3 模型输出 `backbone_feate` |
| `visualizations/dinov3_heatmaps/` | 默认保存目录 |

---

## 🎓 进阶用法

### 批量生成所有样本的热力图

```python
from engine.solver.visualize_features import visualize_dinov3_features

# 生成 batch 中前 4 个样本的热力图
visualize_dinov3_features(
    backbone_feats=backbone_feats,
    images=images,
    max_samples=4,  # ← 可视化 4 个样本
    epoch=epoch,
    step=step
)
```

### 对比不同 Epoch 的特征演化

```bash
# 查看特征变化
ls visualizations/dinov3_heatmaps/
epoch000_step00000_sample0.png  # Epoch 0
epoch010_step05000_sample0.png  # Epoch 10
epoch050_step25000_sample0.png  # Epoch 50
```

---

## 📧 支持

如有问题或建议，请在代码中搜索 `visualize_features.py` 查看完整实现。

