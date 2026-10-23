"""
Transformer Decoder
====================
标准 Pre-LN DETR Decoder：
  N 层 DecoderLayer，每层 = Self-Attn + Cross-Attn + FFN

Object queries 经过 decoder 后，其特征送入：
  1. QNorm-Obj 适配器 → 分类 logits + salience logit
  2. Box MLP          → 边界框预测

LoRA 适配器由外部 attach_incremental_lora() 附加，无需在此特殊处理。
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadSelfAttention, MultiHeadCrossAttention


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    Pre-LN Transformer Decoder Layer:
        q = q + SelfAttn(LN(q))
        q = q + CrossAttn(LN(q), memory)
        q = q + FFN(LN(q))
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

        # Self-attention on object queries
        self.self_attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        # Cross-attention: query ← encoder memory
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout)

        # FFN — LoRA 将替换这两个 Linear
        self.linear1  = nn.Linear(embed_dim, ffn_dim)
        self.linear2  = nn.Linear(ffn_dim,   embed_dim)

        self.norm1    = nn.LayerNorm(embed_dim)
        self.norm2    = nn.LayerNorm(embed_dim)
        self.norm3    = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.act = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query   : (B, Q, D)  object queries
        memory  : (B, N, D)  encoder output
        query_mask : (Q, Q)  causal / self-attn mask (usually None for detection)
        memory_key_padding_mask : (B, N) bool

        Returns
        -------
        (B, Q, D)  updated object queries
        """
        # Self-Attention
        q2 = self.self_attn(self.norm1(query), attn_mask=query_mask)
        query = query + self.dropout1(q2)

        # Cross-Attention
        q2 = self.cross_attn(
            self.norm2(query),
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        query = query + self.dropout2(q2)

        # FFN
        q2    = self.linear2(self.dropout3(self.act(self.linear1(self.norm3(query)))))
        query = query + self.dropout4(q2)
        return query


# ---------------------------------------------------------------------------
# Full Decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """
    N 层堆叠的 DETR Transformer Decoder。

    Parameters
    ----------
    embed_dim   : 特征维度
    num_heads   : 注意力头数
    ffn_dim     : FFN 中间维度
    num_layers  : decoder 层数
    num_queries : object query 数量（DETR 默认 300）
    dropout     : dropout
    return_intermediate : bool
        True → 返回每一层 decoder 的输出（用于 auxiliary losses）
        False → 只返回最后一层输出
    """

    def __init__(
        self,
        embed_dim: int   = 256,
        num_heads: int   = 8,
        ffn_dim: int     = 1024,
        num_layers: int  = 6,
        num_queries: int = 300,
        dropout: float   = 0.1,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.return_intermediate = return_intermediate

        # Learnable object queries: 随机初始化，与 encoder memory 交互
        self.query_embed = nn.Embedding(num_queries, embed_dim)

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        memory : (B, N, D)
            Encoder output.
        memory_key_padding_mask : (B, N) bool

        Returns
        -------
        hs : Tensor
            If return_intermediate=False: (B, Q, D)
            If return_intermediate=True:  (num_layers, B, Q, D)
        """
        B = memory.shape[0]

        # 广播 object queries 到 batch
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)   # (B, Q, D)

        intermediates = []
        for layer in self.layers:
            query = layer(query, memory, memory_key_padding_mask=memory_key_padding_mask)
            if self.return_intermediate:
                intermediates.append(self.norm(query))

        if self.return_intermediate:
            return torch.stack(intermediates, dim=0)   # (L, B, Q, D)

        return self.norm(query)   # (B, Q, D)


# ---------------------------------------------------------------------------
# Box Prediction MLP
# ---------------------------------------------------------------------------

class BoxMLP(nn.Module):
    """
    3 层 MLP 预测边界框 (cx, cy, w, h)，输出经过 sigmoid 归一化到 [0,1]。

    参考 DETR 原始实现。
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256, output_dim: int = 4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) → (..., 4), values in [0,1]"""
        return torch.sigmoid(self.layers(x))
