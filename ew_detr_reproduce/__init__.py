"""
EW-DETR Reproduction
=====================
Three core modules from:
  "EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer"
  Monga et al., Sony Research India / IIIT Hyderabad.

Modules
-------
Module 1 — Incremental Dual LoRA Adapters          (modules/incremental_lora.py)
Module 2 — Query-Norm Objectness Adapter            (modules/qnorm_obj.py)
Module 3 — Unseen Phenotype Discovery               (modules/unseen_phenotype.py)
            (adapted from EUMix — Entropy-Aware Unknown Mixing)

Quick-start
-----------
See tests/test_modules.py for end-to-end examples.
"""

from .modules import (
    IncrementalLoRALinear,
    attach_incremental_lora,
    collect_task_parameters,
    merge_all_task_adapters,
    compute_beta,
    QNormObjectnessAdapter,
    UnseenPhenotypeDiscovery,
    PhenotypeDiscoveryHead,
)
from .models import EWDETRWrapper, StandaloneEWPhenotypeModel
from .utils import compute_all_scores, count_trainable_parameters

__all__ = [
    "IncrementalLoRALinear",
    "attach_incremental_lora",
    "collect_task_parameters",
    "merge_all_task_adapters",
    "compute_beta",
    "QNormObjectnessAdapter",
    "UnseenPhenotypeDiscovery",
    "PhenotypeDiscoveryHead",
    "EWDETRWrapper",
    "StandaloneEWPhenotypeModel",
    "compute_all_scores",
    "count_trainable_parameters",
]
