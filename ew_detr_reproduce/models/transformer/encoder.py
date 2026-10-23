"""
Transformer Encoder
====================
标准 Pre-LN Transformer Encoder：
  N 层 EncoderLayer，每层 = Self-Attn + FFN

LoRA 适配器由外部 attach_incremental_lora() 附加到所有 nn.Linear，
因此此处保持纯标准实现。

位置编码
---------
支持两种：
  - Sinusoidal 2D (默认，无参数)
  - Learnable    (可学习 embedding)
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadSelfAttention


# ---------------------------------------------------------------------------
# 位置编码
# ---------------------------------------------------------------------------

class SinusoidalPosEmbed2D(nn.Module):
    """
    2D 正弦位置编码，适用于展平后的空间 token 序列。

    Parameters
    ----------
    embed_dim     : 特征维度（须为偶数）
    temperature   : 正弦波频率温度
    normalize     : 是否归一化坐标
    """

    def __init__(
        self,
        embed_dim: int,
        temperature: float = 10000.0,
        normalize: bool = True,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim 必须为偶数"
        self.embed_dim   = embed_dim
        self.temperature = temperature
        self.normalize   = normalize

    @torch.no_grad()
    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        返回 (1, h*w, embed_dim) 的位置编码张量。
        """
        D = self.embed_dim // 2   # 行/列各占一半

        y_pos = torch.arange(h, device=device, dtype=torch.float32)
        x_pos = torch.arange(w, device=device, dtype=torch.float32)

        if self.normalize:
            y_pos = y_pos / (h - 1 + 1e-6) * 2 * math.pi
            x_pos = x_pos / (w - 1 + 1e-6) * 2 * math.pi

        dim_t = torch.arange(D, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / D)

        pos_y = y_pos.unsqueeze(1) / dim_t   # (h, D)
        pos_x = x_pos.unsqueeze(1) / dim_t   # (w, D)

        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)  # (h, D)
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)  # (w, D)

        # Outer combination: (h, w, 2D)
        grid_y = pos_y.unsqueeze(1).expand(h, w, D)
        grid_x = pos_x.unsqueeze(0).expand(h, w, D)
        pos = torch.cat([grid_y, grid_x], dim=-1)          # (h, w, embed_dim)
        return pos.reshape(1, h * w, self.embed_dim)        # (1, N, D)


class LearnablePosEmbed(nn.Module):
    """可学习位置编码 (固定序列长度)。"""

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(max_len, embed_dim)

    def forward(self, n: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(n, device=device)
        return self.embed(idx).unsqueeze(0)   # (1, n, D)


# ---------------------------------------------------------------------------
# Encoder Layer
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """
    Pre-LN Transformer Encoder Layer:
        x = x + SelfAttn(LN(x))
        x = x + FFN(LN(x))

    所有 nn.Linear（q/k/v/out_proj, ffn_linear1/2）将被 LoRA 替换。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # FFN
        self.linear1  = nn.Linear(embed_dim, ffn_dim)
        self.linear2  = nn.Linear(ffn_dim,   embed_dim)
        self.norm1    = nn.LayerNorm(embed_dim)
        self.norm2    = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.act = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-Attention (Pre-LN)
        src2 = self.self_attn(self.norm1(src), key_padding_mask=src_key_padding_mask)
        src  = src + self.dropout1(src2)

        # FFN (Pre-LN)
        src2 = self.linear2(self.dropout2(self.act(self.linear1(self.norm2(src)))))
        src  = src + self.dropout3(src2)
        return src


# ---------------------------------------------------------------------------
# Full Encoder
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """
    N 层堆叠的 Transformer Encoder。

    Parameters
    ----------
    embed_dim  : 特征维度
    num_heads  : 注意力头数
    ffn_dim    : FFN 中间维度（通常 = 4 * embed_dim）
    num_layers : encoder 层数
    dropout    : dropout
    pos_type   : 'sinusoidal' | 'learnable'
    max_len    : 若 pos_type='learnable'，最大序列长度
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int   = 1024,
        num_layers: int = 6,
        dropout: float  = 0.1,
        pos_type: str   = "sinusoidal",
        max_len: int    = 4096,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if pos_type == "sinusoidal":
            self.pos_embed = SinusoidalPosEmbed2D(embed_dim)
        else:
            self.pos_embed = LearnablePosEmbed(max_len, embed_dim)

        self.pos_type  = pos_type
        self.embed_dim = embed_dim

    def forward(
        self,
        src: torch.Tensor,
        spatial_shape: Optional[tuple] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src           : (B, N, D)  输入 token 序列
        spatial_shape : (h, w)     仅 sinusoidal 位置编码需要
        src_key_padding_mask : (B, N) bool

        Returns
        -------
        (B, N, D)  encoder memory
        """
        B, N, D = src.shape

        # 加入位置编码
        if self.pos_type == "sinusoidal":
            assert spatial_shape is not None, "sinusoidal 位置编码需要 spatial_shape"
            h, w = spatial_shape
            pos = self.pos_embed(h, w, src.device)      # (1, h*w, D)
        else:
            pos = self.pos_embed(N, src.device)         # (1, N, D)

        x = src + pos

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return self.norm(x)
