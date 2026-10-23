"""
EW-DETR: Integrated Model
==========================
Combines the three modules into a single EWOD-capable wrapper that can be
dropped onto any DETR-family backbone.

Architecture overview (Figure 2 / Figure 3 of the paper)
---------------------------------------------------------

  Image
    │
    ▼
  Frozen Backbone (e.g. ResNet-50 / DINOv2)
    │
    ▼
  Transformer Encoder  ←──── LoRA Adapters (agg frozen + task trainable)
    │
    ▼
  Transformer Decoder  ←──── LoRA Adapters (agg frozen + task trainable)
    │
    ▼  (decoder query features H, shape B × Q × D)
  QNorm-Obj Adapter
    ├── h_cls  (mixed direction vector)
    ├── z_cls  (known + unknown logits)
    └── z_obj  (salience / objectness logit)
    │
    ▼
  Unseen Phenotype / EUMix
    └── z_final  (calibrated logits, shape B × Q × |K_t|+1)

  Localisation head (standard MLP) → bounding boxes
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..modules.incremental_lora import (
    IncrementalLoRALinear,
    attach_incremental_lora,
    collect_task_parameters,
    merge_all_task_adapters,
)
from ..modules.unseen_phenotype import PhenotypeDiscoveryHead


class EWDETRWrapper(nn.Module):
    """
    Generic EWOD wrapper for any DETR-based detector.

    The detector must expose two methods (adapt as needed for your DETR variant):
      ``forward_features(images) -> (hs, memory)``
          Returns decoder output features ``hs`` of shape (B, Q, D) and
          optionally the encoder memory.
      ``forward_boxes(hs) -> boxes``
          Predicts bounding-box coordinates from the decoder features.

    Parameters
    ----------
    detector        : Base DETR model (RF-DETR, Deformable-DETR, etc.).
    embed_dim       : Decoder output dimension D.
    num_known_classes: |K_1| (classes in the first task).
    lora_rank       : Rank r for both LoRA adapters.
    lora_alpha      : LoRA alpha scaling factor.
    lora_dropout    : Dropout on LoRA paths.
    target_modules  : Tuple of layer-name substrings to wrap with LoRA
                      (e.g. ('q_proj', 'k_proj', 'v_proj', 'out_proj')).
                      None → wrap all nn.Linear layers.
    obj_hidden_dim  : Hidden size in the salience MLP.
    beta_min, beta_max : Bounds for the data-aware merging coefficient.
    """

    def __init__(
        self,
        detector: nn.Module,
        embed_dim: int,
        num_known_classes: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: Optional[Tuple[str, ...]] = None,
        obj_hidden_dim: int = 64,
        beta_min: float = 0.1,
        beta_max: float = 0.9,
        init_suppression_strength: float = 0.5,
        learn_suppression: bool = True,
        trainable_detector_modules: Tuple[str, ...] = ("box_head", "bbox_embed", "neck"),
    ):
        super().__init__()
        self.embed_dim          = embed_dim
        self.beta_min           = beta_min
        self.beta_max           = beta_max
        self._task_idx          = 1          # 1-based
        self._n_samples_history = 0          # cumulative samples seen
        self.trainable_detector_modules = trainable_detector_modules

        # 1. Freeze backbone; attach LoRA to encoder + decoder linears
        self._freeze_backbone(detector)
        self.detector = attach_incremental_lora(
            detector,
            rank=lora_rank,
            lora_alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
        )

        # 2. PhenotypeDiscoveryHead (QNorm-Obj + UnseenPhenotype)
        self.head = PhenotypeDiscoveryHead(
            embed_dim=embed_dim,
            num_known_classes=num_known_classes,
            obj_hidden_dim=obj_hidden_dim,
            init_suppression_strength=init_suppression_strength,
            learn_suppression=learn_suppression,
        )
        self._unfreeze_detector_modules()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images : Tensor (B, C, H, W)

        Returns
        -------
        dict with:
            'logits' : calibrated class logits (B, Q, |K_t|+1)
            'z_sal'  : salience logits          (B, Q, 1)
            'boxes'  : predicted boxes           (B, Q, 4)
        """
        hs = self._get_decoder_features(images, *args, **kwargs)   # (B, Q, D)
        boxes = self._get_boxes(hs)                                 # (B, Q, 4)

        z_final, z_sal, _ = self.head(hs)

        return {"logits": z_final, "z_sal": z_sal, "boxes": boxes}

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def on_task_end(self, n_samples_current: int):
        """
        Called at the end of every task (before loading the next task's data).

        1. Merges the task-specific LoRA into the aggregate (data-aware).
        2. Updates task index and cumulative sample count.
        """
        beta_t = merge_all_task_adapters(
            self.detector,
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
        Called at the start of each new task.

        1. Extends the classification head to include new classes.
        2. Unfreezes the task-specific LoRA parameters.
        """
        if num_new_classes < 0:
            raise ValueError("num_new_classes must be non-negative.")
        if num_new_classes > 0:
            self.head.extend_classes(num_new_classes)
        for module in self.detector.modules():
            if isinstance(module, IncrementalLoRALinear):
                module.unfreeze_task()

    def trainable_parameters(self):
        """Return parameter groups for the optimizer."""
        lora_params   = collect_task_parameters(self.detector)
        head_params   = list(self.head.parameters())
        detector_params = self._collect_trainable_detector_parameters()
        return lora_params + head_params + detector_params

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze_backbone(model: nn.Module):
        """Freeze all parameters; LoRA attachment will unfreeze task adapters."""
        for p in model.parameters():
            p.requires_grad_(False)

    def _unfreeze_detector_modules(self):
        for module_name in self.trainable_detector_modules:
            module = getattr(self.detector, module_name, None)
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad_(True)

    def _collect_trainable_detector_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for module_name in self.trainable_detector_modules:
            module = getattr(self.detector, module_name, None)
            if module is None:
                continue
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    # ------------------------------------------------------------------
    # Stubs — override for your DETR variant
    # ------------------------------------------------------------------

    def _get_decoder_features(self, images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Extract last-layer decoder output features.

        Override this method for your specific DETR variant.
        Default: assumes ``self.detector`` has a ``forward_features`` method.
        """
        if hasattr(self.detector, "forward_features"):
            hs, _ = self.detector.forward_features(images, *args, **kwargs)
            return hs
        raise NotImplementedError(
            "Override `_get_decoder_features` for your DETR variant, or "
            "ensure `self.detector.forward_features(images)` returns (hs, memory)."
        )

    def _get_boxes(self, hs: torch.Tensor) -> torch.Tensor:
        """
        Predict bounding boxes from decoder features.

        Override this method for your specific DETR variant.
        Default: assumes ``self.detector`` has a ``forward_boxes`` method.
        """
        if hasattr(self.detector, "forward_boxes"):
            return self.detector.forward_boxes(hs)
        raise NotImplementedError(
            "Override `_get_boxes` for your DETR variant, or ensure "
            "`self.detector.forward_boxes(hs)` is implemented."
        )


# ---------------------------------------------------------------------------
# Standalone (detector-free) version for pure feature-space phenotype discovery
# ---------------------------------------------------------------------------

class StandaloneEWPhenotypeModel(nn.Module):
    """
    Incremental Open-World Phenotype Classifier.

    Useful when no object-detection backbone is needed — e.g. single-cell RNA,
    embedding spaces from pathology encoders, etc.

    The model receives a batch of feature vectors (already encoded) and outputs
    calibrated known-phenotype + unseen-phenotype logits, while maintaining
    incremental LoRA knowledge across tasks.

    Parameters
    ----------
    feature_dim      : Dimensionality of input feature vectors.
    num_known_classes: Number of known phenotypes in the first task.
    lora_rank        : LoRA rank.
    lora_alpha       : LoRA alpha.
    hidden_dim       : Hidden size in the feature projection MLP.
    """

    def __init__(
        self,
        feature_dim: int,
        num_known_classes: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        hidden_dim: int = 256,
        obj_hidden_dim: int = 64,
        beta_min: float = 0.1,
        beta_max: float = 0.9,
        init_suppression_strength: float = 0.5,
        learn_suppression: bool = True,
    ):
        super().__init__()
        self.beta_min           = beta_min
        self.beta_max           = beta_max
        self._task_idx          = 1
        self._n_samples_history = 0

        # Feature projection (acts as a lightweight "encoder")
        base_proj = nn.Linear(feature_dim, hidden_dim)
        self.proj = IncrementalLoRALinear.from_linear(
            base_proj, rank=lora_rank, lora_alpha=lora_alpha
        )

        # Phenotype discovery head
        self.head = PhenotypeDiscoveryHead(
            embed_dim=hidden_dim,
            num_known_classes=num_known_classes,
            obj_hidden_dim=obj_hidden_dim,
            init_suppression_strength=init_suppression_strength,
            learn_suppression=learn_suppression,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (B, feature_dim) or (B, N, feature_dim)

        Returns
        -------
        dict with 'logits' and 'z_sal'.
        """
        h = self.proj(x)                      # (B, hidden_dim)
        z_final, z_sal, _ = self.head(h)
        return {"logits": z_final, "z_sal": z_sal}

    def on_task_end(self, n_samples_current: int) -> float:
        beta_t = merge_all_task_adapters(
            self,
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
            self.head.extend_classes(num_new_classes)
        self.proj.unfreeze_task()

    def trainable_parameters(self):
        return self.proj.task_parameters() + list(self.head.parameters())
