"""
单物种语义原型状态管理器（队列版本）

核心改进：
- 使用环形队列存储特征，不断加入新特征、丢弃旧特征
- 原型 = 队列中所有特征的平均值
- 解决 EMA 的问题：旧的低质量特征影响永远存在

优势：
1. 训练初期的低质量特征会被完全丢弃
2. 原型始终由最近 N 个特征组成，反映模型当前能力
3. 不需要调整 momentum 参数，只需设置队列大小
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


class ProtoState:
    """
    单物种语义原型状态管理器（队列版本）
    
    核心设计：
    - 使用环形缓冲区（Circular Buffer）存储特征
    - 原型 = 队列中所有有效特征的 L2 归一化平均值
    - 队列满时自动丢弃最旧的特征
    
    成员：
    - feature_buffer: Tensor[queue_size, D]（环形缓冲区）
    - proto: Tensor[D]（L2 归一化原型向量）
    - queue_size: int（队列最大容量）
    
    方法：
    - update_sup(feats_gt): 用 GT 框的 embedding 更新队列和原型
    - similarity(feats): 计算 u = (cos + 1) / 2 ∈ [0, 1]
    """
    
    def __init__(
        self,
        feat_dim: int = 256,
        queue_size: int = 1024,  # 队列容量（存储最近 1024 个特征）
        device: torch.device = None,
        # 以下参数保留以保持兼容性（但不再使用）
        tau_sup: float = 256.0,
        tau_unsup: float = 2048.0,
    ):
        self.feat_dim = feat_dim
        self.queue_size = queue_size
        self.device = device or torch.device('cpu')
        
        # 🔥 环形缓冲区（预分配，避免动态增长）
        # 初始化为 None，首次使用时才分配（确保设备正确）
        self.feature_buffer: Optional[torch.Tensor] = None
        
        # 环形缓冲区状态
        self.buffer_ptr: int = 0          # 下一个写入位置
        self.buffer_count: int = 0        # 有效特征数量
        self._buffer_device: Optional[torch.device] = None
        
        # 原型向量（L2 归一化），由队列特征计算得出
        self.proto: Optional[torch.Tensor] = None
        self._proto_gpu: Optional[torch.Tensor] = None
        
        # 统计信息
        self.sup_update_count = 0
        self.unsup_update_count = 0
        self.total_sup_samples = 0
        self.total_unsup_samples = 0
        
        # 🎯 稳定性检测（保留，用于判断原型是否可靠）
        self.proto_change_history: list = []
        self.stability_window: int = 10
        self.stability_threshold: float = 0.05
    
    @staticmethod
    def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        """L2 归一化"""
        return x / (x.norm(dim=dim, keepdim=True) + eps)
    
    def _ensure_buffer(self, device: torch.device, dtype: torch.dtype = torch.float32):
        """确保缓冲区已初始化且在正确的设备上"""
        if self.feature_buffer is None or self._buffer_device != device:
            self.feature_buffer = torch.zeros(
                self.queue_size, self.feat_dim, 
                device=device, dtype=dtype
            )
            self._buffer_device = device
            # 如果是新初始化，重置指针
            if self._buffer_device != device:
                self.buffer_ptr = 0
                self.buffer_count = 0
    
    def _add_features_to_queue(self, feats: torch.Tensor) -> int:
        """
        将特征添加到环形队列
        
        Args:
            feats: 特征向量 [N, D]（已 L2 归一化）
        
        Returns:
            int: 实际添加的特征数量
        """
        device = feats.device
        dtype = feats.dtype
        N = feats.shape[0]
        
        if N == 0:
            return 0
        
        # 确保缓冲区存在
        self._ensure_buffer(device, dtype)
        
        # L2 归一化（确保特征单位化）
        feats_norm = self.l2norm(feats.detach(), dim=1)
        
        # 🔥 向量化写入：批量写入环形缓冲区（避免 Python 循环）
        if N <= self.queue_size:
            # 情况1：添加的特征数量小于队列容量
            # 计算写入位置
            end_ptr = (self.buffer_ptr + N) % self.queue_size
            
            if end_ptr > self.buffer_ptr:
                # 不需要回绕
                self.feature_buffer[self.buffer_ptr:end_ptr] = feats_norm
            else:
                # 需要回绕
                first_part = self.queue_size - self.buffer_ptr
                self.feature_buffer[self.buffer_ptr:] = feats_norm[:first_part]
                if end_ptr > 0:
                    self.feature_buffer[:end_ptr] = feats_norm[first_part:]
            
            self.buffer_ptr = end_ptr
            self.buffer_count = min(self.buffer_count + N, self.queue_size)
        else:
            # 情况2：添加的特征数量大于队列容量，只保留最后 queue_size 个
            self.feature_buffer[:] = feats_norm[-self.queue_size:]
            self.buffer_ptr = 0
            self.buffer_count = self.queue_size
        
        return N
    
    def _compute_proto_from_queue(self) -> Optional[torch.Tensor]:
        """
        从队列中计算原型（L2 归一化平均值）
        
        Returns:
            proto: Tensor[D] 或 None（队列为空时）
        """
        if self.buffer_count == 0 or self.feature_buffer is None:
            return None
        
        # 只取有效部分（buffer_count 个特征）
        if self.buffer_count < self.queue_size:
            # 队列未满，取 [0, buffer_count)
            valid_feats = self.feature_buffer[:self.buffer_count]
        else:
            # 队列已满，取全部
            valid_feats = self.feature_buffer
        
        # 计算平均值并 L2 归一化
        proto = valid_feats.mean(dim=0)
        proto = self.l2norm(proto, dim=0)
        
        return proto
    
    def update_sup(self, feats_gt: torch.Tensor) -> Dict[str, float]:
        """
        用标注数据（GT）更新队列和原型
        
        🔥 队列方式：将新特征添加到队列，旧特征自动被丢弃
        
        Args:
            feats_gt: GT 框的 embedding [N, D]
        
        Returns:
            stats: 更新统计信息
        """
        if feats_gt.shape[0] == 0:
            return {'updated': False, 'n_added': 0, 'queue_size': self.buffer_count}
        
        device = feats_gt.device
        
        # 记录旧原型（用于稳定性检测）
        proto_old = self.proto.clone() if self.proto is not None else None
        
        # 🔥 将新特征添加到队列
        n_added = self._add_features_to_queue(feats_gt)
        
        # 🔥 重新计算原型（队列中所有特征的平均值）
        self.proto = self._compute_proto_from_queue()
        self._proto_gpu = self.proto
        
        # 更新统计
        self.sup_update_count += 1
        self.total_sup_samples += n_added
        
        # 🎯 计算原型变化幅度（用于稳定性检测）
        proto_change = 0.0
        if proto_old is not None and self.proto is not None:
            proto_change = float((self.proto - proto_old.to(device)).norm())
            if self.sup_update_count % 10 == 0:
                self.proto_change_history.append(proto_change)
            if len(self.proto_change_history) > self.stability_window:
                self.proto_change_history.pop(0)
        
        return {
            'updated': True,
            'n_added': n_added,
            'queue_size': self.buffer_count,
            'proto_change': proto_change,
        }
    
    def update_unsup(
        self,
        feats_pseudo: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        用高可信伪标签更新队列和原型
        
        🔥 注意：伪标签更新时，可以选择只添加高权重的特征
        
        Args:
            feats_pseudo: 伪标签框的 embedding [N, D]
            weights: 可选的权重 [N]（高权重的特征才添加到队列）
        
        Returns:
            stats: 更新统计信息
        """
        if feats_pseudo.shape[0] == 0:
            return {'updated': False, 'n_added': 0, 'queue_size': self.buffer_count}
        
        device = feats_pseudo.device
        
        # 🔥 如果有权重，只添加高权重的特征（阈值 0.5）
        if weights is not None:
            high_weight_mask = weights >= 0.5
            if high_weight_mask.sum() == 0:
                return {'updated': False, 'n_added': 0, 'queue_size': self.buffer_count}
            feats_pseudo = feats_pseudo[high_weight_mask]
        
        # 记录旧原型
        proto_old = self.proto.clone() if self.proto is not None else None
        
        # 将特征添加到队列
        n_added = self._add_features_to_queue(feats_pseudo)
        
        # 重新计算原型
        self.proto = self._compute_proto_from_queue()
        self._proto_gpu = self.proto
        
        # 更新统计
        self.unsup_update_count += 1
        self.total_unsup_samples += n_added
        
        # 计算原型变化幅度
        proto_change = 0.0
        if proto_old is not None and self.proto is not None:
            proto_change = float((self.proto - proto_old.to(device)).norm())
        
        return {
            'updated': True,
            'n_added': n_added,
            'queue_size': self.buffer_count,
            'proto_change': proto_change,
        }
    
    def similarity(self, feats: torch.Tensor) -> Optional[torch.Tensor]:
        """
        计算特征与原型的相似度
        
        u = (cos(f, proto) + 1) / 2 ∈ [0, 1]
        
        Args:
            feats: 特征向量 [N, D]
        
        Returns:
            u: 相似度 [N]，范围 [0, 1]
            若原型未初始化则返回 None
        """
        if self.proto is None:
            return None
        
        if feats.shape[0] == 0:
            return torch.tensor([], device=feats.device, dtype=torch.float32)
        
        device = feats.device
        
        # L2 归一化
        feats_norm = self.l2norm(feats, dim=1)  # [N, D]
        
        # 获取原型（确保在正确设备上）
        if self._proto_gpu is not None and self._proto_gpu.device == device:
            proto = self._proto_gpu
        else:
            proto = self.proto.to(device)
            self._proto_gpu = proto
        
        # 余弦相似度
        cos_sim = torch.mm(feats_norm, proto.unsqueeze(1)).squeeze(1)  # [N]
        
        # 映射到 [0, 1]: u = (cos + 1) / 2
        u = (cos_sim + 1) / 2
        
        return u
    
    def is_initialized(self) -> bool:
        """原型是否已初始化（队列中有特征）"""
        return self.buffer_count > 0 and self.proto is not None
    
    def is_stable(self) -> bool:
        """
        判断原型是否已稳定
        
        条件：
        1. 队列已有足够特征（>= queue_size / 4）
        2. 已有足够的更新历史
        3. 最近的变化幅度均值 < stability_threshold
        """
        if self.proto is None:
            return False
        
        # 队列至少有 1/4 满
        if self.buffer_count < self.queue_size // 4:
            return False
        
        if len(self.proto_change_history) < self.stability_window // 2:
            return False
        
        recent_changes = self.proto_change_history[-self.stability_window:]
        avg_change = sum(recent_changes) / len(recent_changes)
        
        return avg_change < self.stability_threshold
    
    def get_stability_score(self) -> float:
        """获取稳定性分数 [0, 1]"""
        if self.proto is None or len(self.proto_change_history) == 0:
            return 0.0
        
        recent_changes = self.proto_change_history[-self.stability_window:]
        avg_change = sum(recent_changes) / len(recent_changes)
        stability = max(0.0, min(1.0, 1.0 - avg_change / self.stability_threshold))
        
        return stability
    
    def get_queue_fill_ratio(self) -> float:
        """获取队列填充比例 [0, 1]"""
        return self.buffer_count / self.queue_size
    
    def get_status_string(self) -> str:
        """获取状态字符串（用于日志）"""
        if self.proto is None:
            return f"proto=None (queue={self.buffer_count}/{self.queue_size})"
        
        proto_norm = float(self.proto.norm()) if self.proto.numel() > 0 else 0.0
        fill_ratio = self.get_queue_fill_ratio()
        
        return (f"proto=init(queue={self.buffer_count}/{self.queue_size}={fill_ratio:.1%}, "
                f"sup={self.sup_update_count}|{self.total_sup_samples}, "
                f"norm={proto_norm:.4f})")
    
    def state_dict(self) -> Dict:
        """保存状态（用于 checkpoint）"""
        return {
            'feature_buffer': self.feature_buffer.clone() if self.feature_buffer is not None else None,
            'buffer_ptr': self.buffer_ptr,
            'buffer_count': self.buffer_count,
            'proto': self.proto.clone() if self.proto is not None else None,
            'sup_update_count': self.sup_update_count,
            'unsup_update_count': self.unsup_update_count,
            'total_sup_samples': self.total_sup_samples,
            'total_unsup_samples': self.total_unsup_samples,
            'feat_dim': self.feat_dim,
            'queue_size': self.queue_size,
            'proto_change_history': self.proto_change_history.copy(),
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """加载状态"""
        # 加载缓冲区
        buffer = state.get('feature_buffer', None)
        if buffer is not None:
            self.feature_buffer = buffer.clone()
            self._buffer_device = buffer.device if isinstance(buffer, torch.Tensor) else None
        
        self.buffer_ptr = state.get('buffer_ptr', 0)
        self.buffer_count = state.get('buffer_count', 0)
        
        # 加载原型
        proto = state.get('proto', None)
        if proto is not None:
            self.proto = proto.clone()
            if isinstance(proto, torch.Tensor) and proto.is_cuda:
                self._proto_gpu = self.proto
        
        # 加载统计信息
        self.sup_update_count = state.get('sup_update_count', 0)
        self.unsup_update_count = state.get('unsup_update_count', 0)
        self.total_sup_samples = state.get('total_sup_samples', 0)
        self.total_unsup_samples = state.get('total_unsup_samples', 0)
        self.proto_change_history = state.get('proto_change_history', [])
        
        # 兼容旧版 checkpoint（如果没有队列信息，从 proto 重建）
        if buffer is None and proto is not None:
            # 旧版只有 proto，没有队列
            # 初始化缓冲区并添加一个虚拟特征（保持接口兼容）
            self.buffer_count = 1
            self.buffer_ptr = 1
