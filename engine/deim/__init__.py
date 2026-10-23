"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


from .deim import DEIM

try:
    # 可选依赖：HungarianMatcher 依赖 scipy（linear_sum_assignment）
    # 训练需要；但为了让部分脚本（如可视化/数据检查）在缺少 scipy 时也能 import，本处做兜底。
    from .matcher import HungarianMatcher  # noqa: F401
except Exception as e:  # noqa: BLE001
    HungarianMatcher = None  # type: ignore
    _HUNGARIAN_MATCHER_IMPORT_ERROR = e

from .hybrid_encoder import HybridEncoder
from .lite_encoder import LiteEncoder


from .dfine_decoder import DFINETransformer
from .rtdetrv2_decoder import RTDETRTransformerv2

from .postprocessor import PostProcessor
from .deim_criterion import DEIMCriterion
from .deim_decoder import DEIMTransformer
