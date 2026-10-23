"""
DINOv2 Backbone — RF-DETR 主干网络
====================================
论文使用 DINOv2-B (ViT-Base/14) 作为 RF-DETR 的 backbone。

DINOv2 特性:
  - patch size = 14，输入 (B,3,H,W) → patch tokens (B, h·w, 768)
  - h = H//14, w = W//14
  - 在 EW-DETR 中 backbone 权重完全冻结，LoRA 只附加到 encoder/decoder

加载方式（两种可选）:
  方式1: torch.hub  —— 需要网络，自动下载
  方式2: timm       —— pip install timm 后可用

输出: 单尺度 patch 特征，形状 (B, N, 768)，N = h·w
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Backbone(nn.Module):
    """
    DINOv2 ViT-Base/14 backbone wrapper.

    Parameters
    ----------
    model_name : str
        DINOv2 变体，支持:
          'dinov2_vits14'  — ViT-Small,  embed_dim=384
          'dinov2_vitb14'  — ViT-Base,   embed_dim=768  ← 论文使用
          'dinov2_vitl14'  — ViT-Large,  embed_dim=1024
          'dinov2_vitg14'  — ViT-Giant,  embed_dim=1536
    use_hub : bool
        True  → 从 torch.hub 加载 (facebookresearch/dinov2)
        False → 从 timm 加载 (需 pip install timm)
    pretrained : bool
        是否加载预训练权重。
    freeze : bool
        True → 冻结所有 backbone 参数（EW-DETR 默认设置）。
    """

    # embed_dim 对应各模型变体
    _EMBED_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        use_hub: bool = True,
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        assert model_name in self._EMBED_DIMS, (
            f"Unknown model: {model_name}. "
            f"Choose from {list(self._EMBED_DIMS.keys())}"
        )
        self.model_name = model_name
        self.embed_dim  = self._EMBED_DIMS[model_name]
        self.patch_size = 14

        # ── 加载模型 ────────────────────────────────────────────────
        if use_hub:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=pretrained,
            )
        else:
            try:
                import timm
                self.dino = timm.create_model(
                    model_name.replace("_", "-"),   # timm 用 '-' 分隔
                    pretrained=pretrained,
                    num_classes=0,                   # 去掉分类头
                )
            except ImportError:
                raise ImportError(
                    "timm 未安装。请运行 `pip install timm` 或设置 use_hub=True。"
                )

        if freeze:
            self.freeze()

    # ------------------------------------------------------------------

    def freeze(self):
        """冻结所有 backbone 参数（EW-DETR 标准做法）。"""
        for p in self.dino.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.dino.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Parameters
        ----------
        x : Tensor (B, 3, H, W)
            输入图像，H 和 W 须为 14 的倍数（如 224, 448, 560, 784…）。

        Returns
        -------
        patch_tokens : Tensor (B, N, embed_dim)
            空间 patch 特征，N = (H//14) × (W//14)。
        spatial_shape : Tuple[int, int]
            (h, w) — 特征图的空间尺寸。
        """
        B, C, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        # DINOv2 forward_features 返回 dict，其中 'x_norm_patchtokens' 是
        # 不含 [CLS] 的 patch 特征 (B, h*w, D)
        feats = self.dino.forward_features(x)

        if isinstance(feats, dict):
            patch_tokens = feats.get(
                "x_norm_patchtokens",
                feats.get("patch_tokens", None)
            )
            if patch_tokens is None:
                # fallback: 取除 CLS token 外的所有 token
                all_tokens = feats.get("x_norm_clstoken", None)
                patch_tokens = feats["x"][:, 1:]   # 去掉 [CLS]
        else:
            # 旧版 DINOv2 直接返回 tensor (B, N+1, D)
            patch_tokens = feats[:, 1:]   # 去掉 [CLS]

        assert patch_tokens.shape == (B, h * w, self.embed_dim), (
            f"DINOv2 输出形状不符: {patch_tokens.shape}, "
            f"期望 ({B}, {h*w}, {self.embed_dim})"
        )
        return patch_tokens, (h, w)


# ---------------------------------------------------------------------------
# 投影 neck: D_backbone → D_model (通常 768 → 256)
# ---------------------------------------------------------------------------

class DINOv2Neck(nn.Module):
    """
    将 DINOv2 的高维 patch 特征线性投影到 DETR 模型维度。

    Parameters
    ----------
    in_dim  : int  DINOv2 输出维度（如 768）
    out_dim : int  DETR 模型维度（如 256）
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_dim) → (B, N, out_dim)"""
        return self.norm(self.proj(x))
