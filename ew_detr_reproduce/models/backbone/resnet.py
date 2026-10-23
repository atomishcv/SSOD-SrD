"""
ResNet-50 Backbone — Deformable-DETR 主干网络
===============================================
论文中 Deformable-DETR 变体使用 ResNet-50 作为 backbone，
输出三个尺度的特征图 (C3, C4, C5)，经过 FPN 投影到统一维度。

输出特征图尺度 (以 800×1333 输入为例):
  C3: stride=8,  (B, 512,  H/8,  W/8)
  C4: stride=16, (B, 1024, H/16, W/16)
  C5: stride=32, (B, 2048, H/32, W/32)

经过投影后统一为 (B, embed_dim, H/s, W/s)，再展平拼接供 encoder 使用。
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet50Backbone(nn.Module):
    """
    torchvision ResNet-50 backbone，提取多尺度特征图。

    Parameters
    ----------
    pretrained : bool
        True → 加载 ImageNet 预训练权重。
    freeze_stages : int
        冻结前 N 个 stage（0=不冻结，1=只冻结 stem，4=全冻结）。
        EW-DETR 中 backbone 完全冻结，默认设为 4。
    return_layers : list
        提取哪些 stage 的特征，默认 ['layer2','layer3','layer4']
        对应 stride [8, 16, 32]。
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_stages: int = 4,
        return_layers: Optional[List[str]] = None,
    ):
        super().__init__()

        try:
            import torchvision.models as tvm
        except ImportError:
            raise ImportError("请安装 torchvision: pip install torchvision")

        # 加载完整 ResNet-50
        if pretrained:
            weights = tvm.ResNet50_Weights.IMAGENET1K_V1
            resnet = tvm.resnet50(weights=weights)
        else:
            resnet = tvm.resnet50(weights=None)

        # 输出通道数 per stage
        self.out_channels: Dict[str, int] = {
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
        }

        if return_layers is None:
            return_layers = ["layer2", "layer3", "layer4"]
        self.return_layers = return_layers

        # 拆分 ResNet 成各子模块
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 冻结指定 stage
        self._freeze_stages(freeze_stages)

    # ------------------------------------------------------------------

    def _freeze_stages(self, n: int):
        stage_map = [
            [self.stem],
            [self.layer1],
            [self.layer2],
            [self.layer3],
            [self.layer4],
        ]
        for i in range(min(n + 1, len(stage_map))):
            for mod in stage_map[i]:
                for p in mod.parameters():
                    p.requires_grad_(False)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (B, 3, H, W)

        Returns
        -------
        dict: {layer_name: feature_map}
            只返回 return_layers 中指定的层。
            每个 feature_map 形状: (B, C, H/stride, W/stride)
        """
        feats = {}
        x = self.stem(x)

        for name, layer in [
            ("layer1", self.layer1),
            ("layer2", self.layer2),
            ("layer3", self.layer3),
            ("layer4", self.layer4),
        ]:
            x = layer(x)
            if name in self.return_layers:
                feats[name] = x

        return feats


# ---------------------------------------------------------------------------
# FPN Neck: 多尺度特征 → 统一维度
# ---------------------------------------------------------------------------

class FPNNeck(nn.Module):
    """
    简单 FPN neck：将每个尺度的特征图分别投影到 embed_dim。

    不做跨尺度融合（Deformable-DETR 原始设计中 encoder 负责跨尺度交互）。

    Parameters
    ----------
    in_channels  : List[int]  各尺度输入通道数（如 [512, 1024, 2048]）
    embed_dim    : int         统一输出维度（如 256）
    """

    def __init__(self, in_channels: List[int], embed_dim: int = 256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, embed_dim),
            )
            for c in in_channels
        ])

    def forward(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        features : dict  {layer_name: (B, C, H, W)}

        Returns
        -------
        List of projected feature maps, each (B, embed_dim, H_s, W_s)
        """
        outs = []
        for proj, feat in zip(self.projs, features.values()):
            outs.append(proj(feat))
        return outs


# ---------------------------------------------------------------------------
# 将多尺度特征展平为序列 (供 Transformer encoder 使用)
# ---------------------------------------------------------------------------

def flatten_multiscale(
    feature_maps: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    将多尺度 2-D 特征图展平并拼接为 1-D token 序列。

    Parameters
    ----------
    feature_maps : List of (B, D, H_s, W_s)

    Returns
    -------
    src          : (B, ΣN_s, D)   — 拼接后的 token 序列
    level_embeds : (B, ΣN_s, 1)   — 标记每个 token 来自第几个尺度
    spatial_shapes : List[(H_s, W_s)] — 每个尺度的空间尺寸
    """
    B = feature_maps[0].shape[0]
    tokens = []
    spatial_shapes = []
    level_ids = []

    for lvl, fmap in enumerate(feature_maps):
        _, D, H, W = fmap.shape
        spatial_shapes.append((H, W))
        # (B, D, H, W) → (B, H*W, D)
        flat = fmap.flatten(2).transpose(1, 2)
        tokens.append(flat)
        level_ids.append(
            torch.full((B, H * W, 1), lvl, dtype=torch.float32, device=fmap.device)
        )

    src = torch.cat(tokens, dim=1)            # (B, ΣN, D)
    level_embeds = torch.cat(level_ids, dim=1)  # (B, ΣN, 1)
    return src, level_embeds, spatial_shapes
