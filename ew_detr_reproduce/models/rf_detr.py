"""
RF-DETR with EW-DETR Modules (主模型)
=======================================
论文主实验模型：RF-DETR + DINOv2-B backbone

完整数据流
----------
  Image (B,3,H,W)
    │
    ▼  [冻结]
  DINOv2-B backbone
    │  patch tokens (B, h·w, 768)
    ▼
  DINOv2Neck (Linear 768→256 + LN)
    │  (B, N, 256)
    ▼
  TransformerEncoder  ← LoRA adapters (agg frozen + task trainable)
    │  memory (B, N, 256)
    ▼
  TransformerDecoder  ← LoRA adapters (agg frozen + task trainable)
    │  query features H (B, Q, 256)
    ▼
  PhenotypeDiscoveryHead
    ├── QNorm-Obj → z_cls (B, Q, |K_t|+1), z_sal (B, Q, 1)
    └── UnseenPhenotype → z_final (B, Q, |K_t|+1)
    │
    ▼
  BoxMLP → boxes (B, Q, 4)

EWOD 生命周期 API
-----------------
  model.on_task_end(n_samples)  — 任务结束：合并 LoRA，重置 task adapter
  model.on_new_task(n_new_cls)  — 新任务开始：扩展分类头，解冻 task adapter
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone.dinov2     import DINOv2Backbone, DINOv2Neck
from .transformer.encoder import TransformerEncoder
from .transformer.decoder import TransformerDecoder, BoxMLP
from ..modules.incremental_lora import (
    IncrementalLoRALinear,
    attach_incremental_lora,
    collect_task_parameters,
    merge_all_task_adapters,
)
from ..modules.unseen_phenotype import PhenotypeDiscoveryHead


class RFDETR_EWOD(nn.Module):
    """
    RF-DETR + EW-DETR 三模块，面向 EWOD 场景的完整模型。

    Parameters
    ----------
    num_known_classes : int
        第一个任务的已知类别数 |K_1|。
    embed_dim         : int
        Transformer 模型维度，默认 256。
    num_queries       : int
        Object query 数量，默认 300。
    num_encoder_layers: int
        Encoder 层数，默认 6。
    num_decoder_layers: int
        Decoder 层数，默认 6。
    num_heads         : int
        注意力头数，默认 8。
    ffn_dim           : int
        FFN 中间维度，默认 1024。
    lora_rank         : int
        LoRA 秩 r，论文使用 8。
    lora_alpha        : float
        LoRA alpha，默认 16（scaling = alpha/rank = 2）。
    lora_target       : tuple
        对哪些层名称附加 LoRA。None 表示所有 Linear 层。
    dinov2_model      : str
        DINOv2 变体，默认 'dinov2_vitb14'（论文使用）。
    dinov2_use_hub    : bool
        True → torch.hub 加载；False → timm 加载。
    beta_min, beta_max: float
        数据感知合并系数的上下界。
    dropout           : float
        全局 dropout，默认 0.1。
    """

    def __init__(
        self,
        num_known_classes: int,
        embed_dim: int          = 256,
        num_queries: int        = 300,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int          = 8,
        ffn_dim: int            = 1024,
        lora_rank: int          = 8,
        lora_alpha: float       = 16.0,
        lora_target: Optional[Tuple[str, ...]] = None,
        dinov2_model: str       = "dinov2_vitb14",
        dinov2_use_hub: bool    = True,
        beta_min: float         = 0.1,
        beta_max: float         = 0.9,
        dropout: float          = 0.1,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.beta_min   = beta_min
        self.beta_max   = beta_max
        self._task_idx  = 1
        self._n_samples_history = 0

        # ── 1. Backbone (冻结) ─────────────────────────────────────────
        self.backbone = DINOv2Backbone(
            model_name=dinov2_model,
            use_hub=dinov2_use_hub,
            pretrained=True,
            freeze=True,                  # backbone 完全冻结
        )
        backbone_dim = self.backbone.embed_dim   # 768 for vitb14

        # ── 2. Neck: backbone_dim → embed_dim ─────────────────────────
        self.neck = DINOv2Neck(in_dim=backbone_dim, out_dim=embed_dim)

        # ── 3. Encoder（将被附加 LoRA）────────────────────────────────
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )

        # ── 4. Decoder（将被附加 LoRA）────────────────────────────────
        self.decoder = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_decoder_layers,
            num_queries=num_queries,
            dropout=dropout,
        )

        # ── 5. Box prediction head ────────────────────────────────────
        self.box_head = BoxMLP(embed_dim=embed_dim, hidden_dim=embed_dim)

        # ── 6. 分类 + 未见表型发现头 ──────────────────────────────────
        self.pheno_head = PhenotypeDiscoveryHead(
            embed_dim=embed_dim,
            num_known_classes=num_known_classes,
        )

        # ── 7. 附加 Incremental LoRA 到 encoder + decoder ─────────────
        #    Neck 和 box_head 不加 LoRA（参数量小，且不涉及任务知识）
        attach_incremental_lora(
            self.encoder,
            rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target,
        )
        attach_incremental_lora(
            self.decoder,
            rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images               : (B, 3, H, W)  输入图像
        src_key_padding_mask : (B, N) bool    encoder padding mask（可选）

        Returns
        -------
        dict:
            'logits' : (B, Q, |K_t|+1)  校准后的分类 logit（含未见表型）
            'boxes'  : (B, Q, 4)         边界框 (cx,cy,w,h) ∈ [0,1]
            'z_sal'  : (B, Q, 1)         salience 分数
        """
        # ── Backbone ──────────────────────────────────────────────────
        patch_tokens, (h, w) = self.backbone(images)     # (B, N, 768)

        # ── Neck ──────────────────────────────────────────────────────
        src = self.neck(patch_tokens)                     # (B, N, 256)

        # ── Encoder ───────────────────────────────────────────────────
        memory = self.encoder(
            src,
            spatial_shape=(h, w),
            src_key_padding_mask=src_key_padding_mask,
        )                                                 # (B, N, 256)

        # ── Decoder ───────────────────────────────────────────────────
        hs = self.decoder(
            memory,
            memory_key_padding_mask=src_key_padding_mask,
        )                                                 # (B, Q, 256)

        # ── 分类 + 未见表型发现 ───────────────────────────────────────
        z_final, z_sal, _ = self.pheno_head(hs)          # (B,Q,|K|+1), (B,Q,1)

        # ── 边界框 ────────────────────────────────────────────────────
        boxes = self.box_head(hs)                         # (B, Q, 4)

        return {"logits": z_final, "boxes": boxes, "z_sal": z_sal}

    # ------------------------------------------------------------------
    # EWOD 任务边界 API
    # ------------------------------------------------------------------

    def on_task_end(self, n_samples_current: int) -> float:
        """
        任务结束时调用：
          1. 数据感知合并 task LoRA → aggregate LoRA
          2. 重置 task LoRA
          3. 更新任务计数器
        """
        beta_t = merge_all_task_adapters(
            nn.ModuleList([self.encoder, self.decoder]),
            task_idx=self._task_idx,
            n_current=n_samples_current,
            n_prev_total=self._n_samples_history,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
        )
        self._n_samples_history += n_samples_current
        self._task_idx += 1
        return beta_t

    def on_new_task(self, num_new_classes: int = 0):
        """
        新任务开始时调用：
          1. 扩展分类头（保留旧类权重）
          2. 解冻 task LoRA 参数
        """
        if num_new_classes < 0:
            raise ValueError("num_new_classes must be non-negative.")
        if num_new_classes > 0:
            self.pheno_head.extend_classes(num_new_classes)
        for mod in list(self.encoder.modules()) + list(self.decoder.modules()):
            if isinstance(mod, IncrementalLoRALinear):
                mod.unfreeze_task()

    # ------------------------------------------------------------------
    # 参数管理
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> List[nn.Parameter]:
        """
        返回当前任务的可训练参数：
          - task-specific LoRA (A, B matrices)
          - pheno_head (QNorm-Obj + UnseenPhenotype)
          - box_head
          - neck
        Backbone 和 aggregate LoRA 保持冻结。
        """
        params = []
        params += collect_task_parameters(self.encoder)
        params += collect_task_parameters(self.decoder)
        params += list(self.pheno_head.parameters())
        params += list(self.box_head.parameters())
        params += list(self.neck.parameters())
        return params

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ------------------------------------------------------------------
    # 便捷构建函数
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        num_known_classes: int,
        lora_rank: int = 8,
        **kwargs,
    ) -> "RFDETR_EWOD":
        """快速构建默认配置的 RF-DETR-EWOD 模型。"""
        return cls(
            num_known_classes=num_known_classes,
            lora_rank=lora_rank,
            **kwargs,
        )
