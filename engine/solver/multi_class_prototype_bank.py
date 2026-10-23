"""
多类别语义原型状态管理器

核心功能：
1. 为每个类别维护一个独立的 L2 归一化原型向量
2. 两阶段更新策略：
   - 阶段A（训练早期）：用 GT 按类别初始化/暖启动各类别原型
   - 阶段B（训练中后期）：用高可信伪标签按类别小步更新原型
3. 计算"类别相似度 u"用于筛掉"错误类别"的伪标签

关键设计：
- 每个类别一个 proto，互相独立
- 动态 momentum：m = exp(-n_eff / tau)
- 筛选时：检查伪标签类别是否与对应原型匹配
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, List


class MultiClassProtoState:
    """
    多类别语义原型状态管理器
    
    成员：
    - protos: Dict[int, Tensor[D]]（每个类别一个 L2 normalized 原型向量）
    - tau_sup: float（GT 更新时间常数）
    - tau_unsup: float（伪标签更新时间常数）
    
    方法：
    - update_sup(feats_gt, labels_gt): 按类别用 GT 更新各类别原型
    - similarity(feats, labels): 计算每个特征与其**对应类别**原型的相似度
    """
    
    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 256,
        tau_sup: float = 256.0,
        tau_unsup: float = 2048.0,
        device: torch.device = None,
    ):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.tau_sup = tau_sup
        self.tau_unsup = tau_unsup
        self.device = device or torch.device('cpu')
        
        # 每个类别一个原型向量
        self.protos: Dict[int, torch.Tensor] = {}
        self._proto_gpu_cache: Dict[int, torch.Tensor] = {}
        
        # 统计信息（按类别）
        self.sup_update_count: Dict[int, int] = {}
        self.unsup_update_count: Dict[int, int] = {}
        self.total_sup_samples: Dict[int, int] = {}
        self.total_unsup_samples: Dict[int, int] = {}
        
        # 稳定性检测（按类别）
        self.proto_change_history: Dict[int, List[float]] = {}
        self.stability_window: int = 10
        self.stability_threshold: float = 0.05
    
    @staticmethod
    def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        """L2 归一化"""
        return x / (x.norm(dim=dim, keepdim=True) + eps)
    
    def _update_proto_for_class(
        self,
        cls_id: int,
        feats: torch.Tensor,
        tau: float,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        更新单个类别的原型
        """
        if feats.shape[0] == 0:
            return {'updated': False, 'n_eff': 0, 'momentum': 0}
        
        device = feats.device
        N = feats.shape[0]
        
        # L2 归一化输入特征
        feats_norm = self.l2norm(feats, dim=1)
        
        # 计算加权平均 μ
        if weights is not None:
            weights = weights.to(device)
            w_sum = weights.sum() + 1e-8
            w_norm = weights / w_sum
            mu = (feats_norm * w_norm.unsqueeze(1)).sum(dim=0)
            n_eff = float(w_sum)
        else:
            mu = feats_norm.mean(dim=0)
            n_eff = float(N)
        
        # L2 归一化 μ
        mu = self.l2norm(mu, dim=0)
        
        proto_change = 0.0
        if cls_id not in self.protos:
            # 第一次初始化
            self.protos[cls_id] = mu.detach()
            self._proto_gpu_cache[cls_id] = self.protos[cls_id]
            self.proto_change_history[cls_id] = []
            self.sup_update_count[cls_id] = 0
            self.unsup_update_count[cls_id] = 0
            self.total_sup_samples[cls_id] = 0
            self.total_unsup_samples[cls_id] = 0
            momentum = 0.0
        else:
            # 动态 momentum
            momentum = math.exp(-n_eff / tau)
            
            proto_old = self.protos[cls_id].to(device).clone()
            proto = self.protos[cls_id].to(device)
            proto = momentum * proto + (1 - momentum) * mu
            proto = self.l2norm(proto, dim=0)
            
            self.protos[cls_id] = proto.detach()
            self._proto_gpu_cache[cls_id] = self.protos[cls_id]
            
            # 计算变化幅度
            proto_change = float((proto - proto_old).norm().detach())
            if self.sup_update_count.get(cls_id, 0) % 100 == 0:
                if cls_id not in self.proto_change_history:
                    self.proto_change_history[cls_id] = []
                self.proto_change_history[cls_id].append(proto_change)
                if len(self.proto_change_history[cls_id]) > self.stability_window:
                    self.proto_change_history[cls_id].pop(0)
        
        return {
            'updated': True,
            'n_eff': n_eff,
            'momentum': momentum,
        }
    
    def update_sup(
        self,
        feats_gt: torch.Tensor,
        labels_gt: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        用标注数据（GT）按类别更新原型
        
        Args:
            feats_gt: GT 框的 embedding [N, D]
            labels_gt: GT 框的类别标签 [N]
        
        Returns:
            stats: 各类别的更新统计信息
        """
        all_stats = {}
        
        if feats_gt.shape[0] == 0:
            return all_stats
        
        # 按类别分组更新
        unique_labels = labels_gt.unique()
        for cls_id in unique_labels:
            cls_id_int = int(cls_id)
            mask = labels_gt == cls_id
            feats_cls = feats_gt[mask]
            
            stats = self._update_proto_for_class(cls_id_int, feats_cls, tau=self.tau_sup)
            if stats['updated']:
                self.sup_update_count[cls_id_int] = self.sup_update_count.get(cls_id_int, 0) + 1
                self.total_sup_samples[cls_id_int] = self.total_sup_samples.get(cls_id_int, 0) + feats_cls.shape[0]
            
            all_stats[cls_id_int] = stats
        
        return all_stats
    
    def update_unsup(
        self,
        feats_pseudo: torch.Tensor,
        labels_pseudo: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        用高可信伪标签按类别更新原型
        """
        all_stats = {}
        
        if feats_pseudo.shape[0] == 0:
            return all_stats
        
        unique_labels = labels_pseudo.unique()
        for cls_id in unique_labels:
            cls_id_int = int(cls_id)
            mask = labels_pseudo == cls_id
            feats_cls = feats_pseudo[mask]
            weights_cls = weights[mask] if weights is not None else None
            
            stats = self._update_proto_for_class(cls_id_int, feats_cls, tau=self.tau_unsup, weights=weights_cls)
            if stats['updated']:
                self.unsup_update_count[cls_id_int] = self.unsup_update_count.get(cls_id_int, 0) + 1
                self.total_unsup_samples[cls_id_int] = self.total_unsup_samples.get(cls_id_int, 0) + feats_cls.shape[0]
            
            all_stats[cls_id_int] = stats
        
        return all_stats
    
    def similarity(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        计算每个特征与其**对应类别**原型的相似度
        
        🔥 关键改进：不是与单一原型比较，而是与对应类别的原型比较
        
        Args:
            feats: 特征向量 [N, D]
            labels: 类别标签 [N]（每个特征对应的类别）
        
        Returns:
            u: 相似度 [N]，范围 [0, 1]
            若对应类别原型未初始化则返回 1.0（通过筛选）
        """
        if feats.shape[0] == 0:
            return torch.tensor([], device=feats.device, dtype=torch.float32)
        
        device = feats.device
        N = feats.shape[0]
        
        # L2 归一化
        feats_norm = self.l2norm(feats, dim=1)  # [N, D]
        
        # 逐个计算相似度
        u = torch.ones(N, device=device, dtype=torch.float32)
        
        for i in range(N):
            cls_id = int(labels[i])
            
            if cls_id not in self.protos:
                # 该类别原型未初始化，默认通过（u=1）
                continue
            
            proto = self._proto_gpu_cache.get(cls_id, self.protos[cls_id]).to(device)
            
            # 余弦相似度
            cos_sim = torch.dot(feats_norm[i], proto)
            
            # 映射到 [0, 1]
            u[i] = (cos_sim + 1) / 2
        
        return u
    
    def similarity_batch(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        批量计算相似度（优化版本，减少 Python 循环）
        """
        if feats.shape[0] == 0:
            return torch.tensor([], device=feats.device, dtype=torch.float32)
        
        device = feats.device
        N = feats.shape[0]
        
        # L2 归一化
        feats_norm = self.l2norm(feats, dim=1)  # [N, D]
        
        # 初始化结果（默认 1.0 表示通过）
        u = torch.ones(N, device=device, dtype=torch.float32)
        
        # 按类别批量处理
        unique_labels = labels.unique()
        for cls_id in unique_labels:
            cls_id_int = int(cls_id)
            
            if cls_id_int not in self.protos:
                continue
            
            mask = labels == cls_id
            feats_cls = feats_norm[mask]  # [M, D]
            
            proto = self._proto_gpu_cache.get(cls_id_int, self.protos[cls_id_int]).to(device)
            
            # 批量余弦相似度
            cos_sim = torch.mm(feats_cls, proto.unsqueeze(1)).squeeze(1)  # [M]
            
            # 映射到 [0, 1]
            u[mask] = (cos_sim + 1) / 2
        
        return u
    
    def is_initialized(self) -> bool:
        """至少有一个类别的原型已初始化"""
        return len(self.protos) > 0
    
    def is_class_initialized(self, cls_id: int) -> bool:
        """指定类别的原型是否已初始化"""
        return cls_id in self.protos
    
    def is_stable(self) -> bool:
        """所有已初始化的类别原型是否都已稳定"""
        if len(self.protos) == 0:
            return False
        
        for cls_id in self.protos:
            history = self.proto_change_history.get(cls_id, [])
            if len(history) < self.stability_window // 2:
                return False
            
            recent_changes = history[-self.stability_window:]
            avg_change = sum(recent_changes) / len(recent_changes)
            if avg_change >= self.stability_threshold:
                return False
        
        return True
    
    def get_status_string(self) -> str:
        """获取状态字符串（用于日志）"""
        if len(self.protos) == 0:
            return "MultiClassProto: no classes initialized"
        
        parts = []
        for cls_id in sorted(self.protos.keys()):
            sup_cnt = self.sup_update_count.get(cls_id, 0)
            sup_samples = self.total_sup_samples.get(cls_id, 0)
            parts.append(f"cls{cls_id}(sup={sup_cnt}|{sup_samples})")
        
        return f"MultiClassProto: {', '.join(parts)}"
    
    def state_dict(self) -> Dict:
        """保存状态"""
        return {
            'protos': {k: v.clone() for k, v in self.protos.items()},
            'sup_update_count': dict(self.sup_update_count),
            'unsup_update_count': dict(self.unsup_update_count),
            'total_sup_samples': dict(self.total_sup_samples),
            'total_unsup_samples': dict(self.total_unsup_samples),
            'num_classes': self.num_classes,
            'feat_dim': self.feat_dim,
            'tau_sup': self.tau_sup,
            'tau_unsup': self.tau_unsup,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """加载状态"""
        protos = state.get('protos', {})
        self.protos = {k: v.clone() for k, v in protos.items()}
        self._proto_gpu_cache = {}
        for k, v in self.protos.items():
            if v.is_cuda:
                self._proto_gpu_cache[k] = v
        
        self.sup_update_count = dict(state.get('sup_update_count', {}))
        self.unsup_update_count = dict(state.get('unsup_update_count', {}))
        self.total_sup_samples = dict(state.get('total_sup_samples', {}))
        self.total_unsup_samples = dict(state.get('total_unsup_samples', {}))
