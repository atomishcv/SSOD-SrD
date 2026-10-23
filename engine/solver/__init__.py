"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from ._solver import BaseSolver
from .clas_solver import ClasSolver
from .det_solver import DetSolver
from .ssod_solver import SSODSolver
try:
    # 可选：部分脚本/环境不需要无监督损失或缺少 scipy（HungarianMatcher），此处做兜底避免 import 失败
    from .unsup_loss_computer import compute_unsup_losses  # noqa: F401
except Exception:  # noqa: BLE001
    compute_unsup_losses = None  # type: ignore

try:
    from .ssod_losses import (  # noqa: F401
    compute_sup_loss, 
    compute_unsup_loss, 
    compute_unsup_loss_with_consistency,
    compute_unsup_loss_with_roi_consistency,
        weighted_loss,
)
except Exception:  # noqa: BLE001
    compute_sup_loss = None  # type: ignore
    compute_unsup_loss = None  # type: ignore
    compute_unsup_loss_with_consistency = None  # type: ignore
    compute_unsup_loss_with_roi_consistency = None  # type: ignore
    weighted_loss = None  # type: ignore

try:
    from .consistency_loss import (  # noqa: F401
    ConsistencyLoss,
    compute_query_consistency_loss,
    compute_feature_consistency_loss,
)
except Exception:  # noqa: BLE001
    ConsistencyLoss = None  # type: ignore
    compute_query_consistency_loss = None  # type: ignore
    compute_feature_consistency_loss = None  # type: ignore

# 🔥 Semi-DETR 核心模块
from .semi_detr_modules import (
    extract_queries_from_pseudo_labels,
    compute_semi_detr_unsup_loss,
)



from typing import Dict

TASKS :Dict[str, BaseSolver] = {
    'classification': ClasSolver,
    'detection': DetSolver,
    'ssod': SSODSolver,
}
