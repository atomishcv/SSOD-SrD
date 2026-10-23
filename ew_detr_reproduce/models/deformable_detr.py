"""
Deformable-DETR with EW-DETR Modules (副模型)
===============================================
论文辅助实验模型：Deformable-DETR + ResNet-50 backbone

完整数据流
----------
  Image (B,3,H,W)
    │
    ▼  [冻结]
  ResNet-50 backbone  →  多尺度特征 {C3,C4,C5}
    │  {(B,512,H/8,W/8), (B,1024,H/16,W/16), (B,2048,H/32,W/32)}
    ▼
  FPNNeck             →  各尺度投影到 256
    │  [(B,256,H/8,W/8), (B,256,H/16,W/16), (B,256,H/32,W/32)]
    ▼
  flatten_multiscale  →  展平拼接 (B, ΣN, 256)
    ▼
  TransformerEncoder  ← LoRA adapters
    │  memory (B, ΣN, 256)
    ▼
  TransformerDecoder  ← LoRA adapters
    │  query features H (B, Q, 256)
    ▼
  PhenotypeDiscoveryHead
    ├── z_final (B, Q, |K_t|+1)
    └── z_sal   (B, Q, 1)
    ▼
  BoxMLP → boxes (B, Q, 4)

注意：此处使用标准多头注意力代替 Deformable Attention，
保留 Deformable-DETR 的多尺度输入特征与整体结构，
如需完整 Deformable Attention 可替换 encoder 中的 attention 模块。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone.resnet      import ResNet50Backbone, FPNNeck, flatten_multiscale
from .transformer.encoder  import TransformerEncoder
from .transformer.decoder  import TransformerDecoder, BoxMLP
from ..modules.incremental_lora import (
    IncrementalLoRALinear,
    attach_incremental_lora,
    collect_task_parameters,
    merge_all_task_adapters,
)
from ..modules.unseen_phenotype import PhenotypeDiscoveryHead


class DeformableDETR_EWOD(nn.Module):
    """
    Deformable-DETR + EW-DETR 三模块，面向 EWOD 场景的完整模型。

    Parameters
    ----------
    num_known_classes  : 第一个任务的已知类别数。
    embed_dim          : Transformer 模型维度，默认 256。
    num_queries        : Object query 数量，默认 300。
    num_encoder_layers : Encoder 层数，默认 6。
    num_decoder_layers : Decoder 层数，默认 6。
    num_heads          : 注意力头数，默认 8。
    ffn_dim            : FFN 中间维度，默认 1024。
    backbone_stages    : ResNet 提取的 stage 列表，默认 ['layer2','layer3','layer4']。
    freeze_backbone_stages : 冻结 ResNet 前 N 个 stage，默认 4（全冻结）。
    lora_rank          : LoRA 秩。
    lora_alpha         : LoRA alpha。
    beta_min, beta_max : 数据感知合并系数范围。
    dropout            : Dropout。
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
        backbone_stages: Optional[List[str]] = None,
        freeze_backbone_stages: int = 4,
        lora_rank: int          = 8,
        lora_alpha: float       = 16.0,
        lora_target: Optional[Tuple[str, ...]] = None,
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

        if backbone_stages is None:
            backbone_stages = ["layer2", "layer3", "layer4"]

        # ResNet 各 stage 输出通道
        stage_channels = {"layer2": 512, "layer3": 1024, "layer4": 2048}
        in_channels = [stage_channels[s] for s in backbone_stages]

        # ── 1. Backbone (冻结) ─────────────────────────────────────────
        self.backbone = ResNet50Backbone(
            pretrained=True,
            freeze_stages=freeze_backbone_stages,
            return_layers=backbone_stages,
        )

        # ── 2. FPN Neck ───────────────────────────────────────────────
        self.neck = FPNNeck(in_channels=in_channels, embed_dim=embed_dim)

        # ── 3. Encoder（将被附加 LoRA）────────────────────────────────
        #  多尺度输入序列长度不固定，使用 learnable 位置编码
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_encoder_layers,
            dropout=dropout,
            pos_type="learnable",
            max_len=8192,          # 足够容纳最大多尺度序列
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

        # ── 7. 尺度级别 embedding（区分多尺度 token 来源）─────────────
        self.level_embed = nn.Embedding(len(backbone_stages), embed_dim)

        # ── 8. 附加 Incremental LoRA ──────────────────────────────────
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
        images : (B, 3, H, W)

        Returns
        -------
        dict: 'logits' (B,Q,|K|+1), 'boxes' (B,Q,4), 'z_sal' (B,Q,1)
        """
        B = images.shape[0]

        # ── Backbone ──────────────────────────────────────────────────
        feat_dict = self.backbone(images)             # {layer_name: (B,C,H,W)}

        # ── FPN Neck ──────────────────────────────────────────────────
        proj_feats = self.neck(feat_dict)             # [(B,256,H/s,W/s)]

        # ── 展平 + 加尺度 embedding ───────────────────────────────────
        src, _, spatial_shapes = flatten_multiscale(proj_feats)   # (B, ΣN, 256)
        for lvl in range(len(proj_feats)):
            h, w = spatial_shapes[lvl]
            n_lvl = h * w
            start = sum(h_ * w_ for h_, w_ in spatial_shapes[:lvl])
            src[:, start:start + n_lvl] += self.level_embed.weight[lvl]

        # ── Encoder ───────────────────────────────────────────────────
        memory = self.encoder(src)                    # (B, ΣN, 256)

        # ── Decoder ───────────────────────────────────────────────────
        hs = self.decoder(memory)                     # (B, Q, 256)

        # ── 分类 + 未见表型发现 ───────────────────────────────────────
        z_final, z_sal, _ = self.pheno_head(hs)

        # ── 边界框 ────────────────────────────────────────────────────
        boxes = self.box_head(hs)

        return {"logits": z_final, "boxes": boxes, "z_sal": z_sal}

    # ------------------------------------------------------------------
    # EWOD 任务边界 API（与 RF-DETR 版本完全一致）
    # ------------------------------------------------------------------

    def on_task_end(self, n_samples_current: int) -> float:
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
        if num_new_classes < 0:
            raise ValueError("num_new_classes must be non-negative.")
        if num_new_classes > 0:
            self.pheno_head.extend_classes(num_new_classes)
        for mod in list(self.encoder.modules()) + list(self.decoder.modules()):
            if isinstance(mod, IncrementalLoRALinear):
                mod.unfreeze_task()

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        params += collect_task_parameters(self.encoder)
        params += collect_task_parameters(self.decoder)
        params += list(self.pheno_head.parameters())
        params += list(self.box_head.parameters())
        params += list(self.neck.parameters())
        params += list(self.level_embed.parameters())
        return params

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    @classmethod
    def build(cls, num_known_classes: int, **kwargs) -> "DeformableDETR_EWOD":
        return cls(num_known_classes=num_known_classes, **kwargs)
