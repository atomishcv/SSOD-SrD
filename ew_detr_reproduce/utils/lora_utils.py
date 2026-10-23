"""
LoRA utility helpers: SVD analysis, parameter counting, weight visualisation.
"""

from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Return a dict with trainable, frozen, and total parameter counts.
    """
    from ..modules.incremental_lora import IncrementalLoRALinear

    trainable_lora = 0
    frozen_lora    = 0
    other          = 0

    for module in model.modules():
        if isinstance(module, IncrementalLoRALinear):
            # Task-specific (trainable)
            trainable_lora += module.task.A.numel() + module.task.B.numel()
            # Aggregate (frozen buffer)
            frozen_lora    += module.agg.A.numel()  + module.agg.B.numel()
            # Base weight (frozen)
            frozen_lora    += module.weight.numel()
            if module.bias is not None:
                frozen_lora += module.bias.numel()
        else:
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    other += p.numel()

    return {
        "trainable_lora":   trainable_lora,
        "frozen_lora":      frozen_lora,
        "other_trainable":  other,
        "total_trainable":  trainable_lora + other,
    }


def effective_rank(matrix: torch.Tensor, threshold: float = 0.99) -> int:
    """
    Estimate the effective rank of a 2-D matrix by finding the minimum number
    of singular values that capture *threshold* fraction of the total variance.
    """
    assert matrix.dim() == 2
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    cumulative = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    rank = int((cumulative < threshold).sum().item()) + 1
    return rank


def spectral_summary(model: nn.Module) -> List[dict]:
    """
    For every IncrementalLoRALinear in the model, return a summary of:
      - layer name
      - effective rank of ΔW_agg
      - effective rank of ΔW_task
      - ratio of the top singular value to the Frobenius norm (condition proxy)
    """
    from ..modules.incremental_lora import IncrementalLoRALinear

    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, IncrementalLoRALinear):
            continue

        dW_agg  = module.agg.delta_weight().detach().float()
        dW_task = module.task.delta_weight().detach().float()

        S_agg  = torch.linalg.svdvals(dW_agg)
        S_task = torch.linalg.svdvals(dW_task)

        rows.append({
            "layer":          name,
            "eff_rank_agg":   effective_rank(dW_agg),
            "eff_rank_task":  effective_rank(dW_task),
            "top_sv_agg":     S_agg[0].item(),
            "frob_agg":       dW_agg.norm().item(),
            "top_sv_task":    S_task[0].item(),
            "frob_task":      dW_task.norm().item(),
        })

    return rows
