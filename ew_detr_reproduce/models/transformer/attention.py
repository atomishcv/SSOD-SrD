"""
Transformer Attention 组件
===========================
实现两种 attention：
  1. MultiHeadSelfAttention  — 标准 MHSA，用于 encoder/decoder self-attn
  2. MultiHeadCrossAttention — 标准 MHCA，用于 decoder cross-attn

EW-DETR 的 LoRA 适配器附加到这些模块内部所有的 nn.Linear 层上，
因此这里保持标准实现（plain nn.Linear），
由外部的 attach_incremental_lora() 统一替换。
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    标准多头自注意力。

    Parameters
    ----------
    embed_dim  : 模型维度 D
    num_heads  : 头数 H
    dropout    : attention dropout
    bias       : QKV 投影是否加 bias
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.dropout    = dropout

        # ── LoRA 将替换这四个 Linear ─────────────────────────────────
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                : (B, N, D)
        key_padding_mask : (B, N) bool, True 处被忽略
        attn_mask        : (N, N) additive mask

        Returns
        -------
        (B, N, D)
        """
        B, N, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        Q = self.q_proj(x).reshape(B, N, H, Hd).transpose(1, 2)   # (B,H,N,Hd)
        K = self.k_proj(x).reshape(B, N, H, Hd).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, H, Hd).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale              # (B,H,N,N)

        if attn_mask is not None:
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # (B,N) → (B,1,1,N)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = (attn @ V).transpose(1, 2).reshape(B, N, D)          # (B,N,D)
        return self.out_proj(out)


class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力（Query 来自 decoder，Key/Value 来自 encoder memory）。

    Parameters
    ----------
    embed_dim  : 模型维度
    num_heads  : 头数
    dropout    : attention dropout
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.dropout   = dropout

        # ── LoRA 将替换这四个 Linear ─────────────────────────────────
        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query   : (B, Q, D)  decoder queries
        memory  : (B, N, D)  encoder output
        memory_key_padding_mask : (B, N) bool

        Returns
        -------
        (B, Q, D)
        """
        B, Nq, D = query.shape
        Nm = memory.shape[1]
        H, Hd = self.num_heads, self.head_dim

        Q = self.q_proj(query).reshape(B, Nq, H, Hd).transpose(1, 2)   # (B,H,Q,Hd)
        K = self.k_proj(memory).reshape(B, Nm, H, Hd).transpose(1, 2)  # (B,H,N,Hd)
        V = self.v_proj(memory).reshape(B, Nm, H, Hd).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale                   # (B,H,Q,N)

        if memory_key_padding_mask is not None:
            mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(2)    # (B,1,1,N)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = (attn @ V).transpose(1, 2).reshape(B, Nq, D)              # (B,Q,D)
        return self.out_proj(out)
