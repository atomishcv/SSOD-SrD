"""
Semi-DETR style semi-supervised object detection for DEIM

核心组件（参考 Semi-DETR）：
1. DEIMSemiDetector: 包装 DEIM 的半监督检测器
2. MeanTeacher: EMA 教师模型更新
3. Projector: RoI 特征投影到 Query 空间
4. GMMFilter: GMM 伪标签筛选
5. MultiBranchPipeline: 弱/强增广数据管道
6. SSODSolver: 半监督训练 Solver
"""

from .deim_ssod import DEIMSemiDetector
from .projector import Projector, MultiScaleRoIExtractor
from .mean_teacher import MeanTeacher
from .ssod_dataset import (
    SemiDataset, 
    MultiBranchPipeline,
    CocoLabeledDataset,
    CocoUnlabeledDataset,
    WeakAugmentation,
    StrongAugmentation,
    SemiBalanceSampler,
    semi_collate_fn,
)
from .ssod_engine import train_one_epoch_ssod, evaluate
from .ssod_solver import SSODSolver
from .gmm_filter import GMMFilter, compute_matching_cost

__all__ = [
    # 模型
    'DEIMSemiDetector',
    'Projector',
    'MultiScaleRoIExtractor',
    
    # EMA
    'MeanTeacher',
    
    # 数据
    'SemiDataset',
    'MultiBranchPipeline',
    'CocoLabeledDataset',
    'CocoUnlabeledDataset',
    'WeakAugmentation',
    'StrongAugmentation',
    'SemiBalanceSampler',
    'semi_collate_fn',
    
    # 训练
    'train_one_epoch_ssod',
    'evaluate',
    'SSODSolver',
    
    # 筛选
    'GMMFilter',
    'compute_matching_cost',
]

