"""
Semi-DETR 核心模块实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import deque

from ..deim.utils import inverse_sigmoid
from ..deim.box_ops import box_cxcywh_to_xyxy, box_iou

if TYPE_CHECKING:
    pass

from .transform_utils import transform_boxes_cxcywh_weak_to_strong


# ============================================================================
# 0. 自适应阈值管理器（GMM + EM 两高斯混合模型）
# ============================================================================

class AdaptiveThresholdManager:
    """
    自适应置信度阈值管理器（对数正态分布 + 95%分位数）
    
    核心思路：
    - 假设 teacher 的置信度分数服从对数正态分布
    - 使用极大似然估计（MLE）计算分布参数（μ, σ）
    - 选择95%分位数作为阈值：高于此阈值视为前景，低于视为背景
    - 使用全局单一阈值（不再按类别分别设定）
    
    数学原理：
    1. 对数正态分布：X ~ LogNormal(μ, σ²)，其中 ln(X) ~ N(μ, σ²)
    2. 极大似然估计：
       - μ = mean(ln(scores))
       - σ² = var(ln(scores))
    3. 95%分位数阈值：τ = exp(μ + 1.645 * σ)
       （1.645 是标准正态分布的95%分位数，即 P(Z ≤ 1.645) = 0.95）
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        window_size: int = 5000,
        min_samples: int = 50,
        quantile: float = 0.95,
        tau_min: float = 0.2,
        tau_max: float = 0.9,
        tau_init: float = 0.45,
        ema_alpha: float = 0.05,
        fallback_quantile: float = 0.7,
        class_names: Optional[List[str]] = None,
        device: torch.device = None,
    ):
        self.num_classes = num_classes
        self.window_size = window_size
        self.min_samples = min_samples
        self.quantile = quantile
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.ema_alpha = ema_alpha
        self.fallback_quantile = fallback_quantile
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.device = device or torch.device('cpu')
        
        self.score_buffer: deque = deque(maxlen=window_size)
        self.tau_init = tau_init
        self.threshold = tau_init
        self.lognormal_params: Optional[Dict] = None
        self.update_count = 0
        self.last_update_info: Dict[str, any] = {}
    
    def _fit_lognormal_mle(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用极大似然估计拟合对数正态分布参数（纯 GPU 操作，无同步）
        
        Args:
            scores: 置信度分数 [N]，范围 (0, 1)
            
        Returns:
            (mu, sigma): 对数正态分布的参数（tensor 格式）
        """
        if scores.numel() < 2:
            device = scores.device
            return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        
        eps = 1e-6
        scores_clamped = scores.clamp(eps, 1.0 - eps)
        log_scores = torch.log(scores_clamped)
        mu = log_scores.mean()
        sigma = torch.sqrt(((log_scores - mu) ** 2).mean()).clamp(min=1e-6)
        
        return mu, sigma
    
    def _compute_threshold_from_lognormal(
        self,
        scores: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据对数正态分布参数计算95%分位数阈值（纯 GPU 操作）
        
        Args:
            scores: 置信度分数 [N]
            mu: 对数正态分布的均值参数（tensor）
            sigma: 对数正态分布的标准差参数（tensor）
            
        Returns:
            threshold: 95%分位数阈值（tensor）
        """
        z_quantile = 1.645
        threshold = torch.exp(mu + z_quantile * sigma)
        threshold = threshold.clamp(self.tau_min, self.tau_max)
        
        return threshold
    
    def update(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> None:
        """
        更新分数缓冲区（周期性收集，减少 GPU/CPU 同步）
        
        Args:
            scores: teacher 的置信度分数 [N]
            labels: 对应的预测类别 [N]（不再使用，保留兼容性）
        """
        if scores.numel() == 0:
            return
        
        self._call_count = getattr(self, '_call_count', 0) + 1
        if self._call_count % 5 != 0:
            return
        
        num_samples = min(scores.numel(), 100)
        if num_samples < scores.numel():
            step = scores.numel() // num_samples
            scores = scores[::step][:num_samples]
        
        scores_cpu = scores.detach().cpu()
        for s in scores_cpu.tolist():
            self.score_buffer.append(s)
    
    def compute_thresholds(self, log: bool = True) -> float:
        """
        基于当前缓冲区计算自适应阈值（对数正态分布 + 95%分位数）
        
        Returns:
            全局单一阈值
        """
        n_samples = len(self.score_buffer)
        
        if n_samples < self.min_samples:
            if n_samples > 0:
                scores_tensor = torch.tensor(list(self.score_buffer))
                sorted_s, _ = torch.sort(scores_tensor)
                q_idx = int(self.fallback_quantile * n_samples)
                tau_new = sorted_s[q_idx].item()
                tau_new = max(self.tau_min, min(self.tau_max, tau_new))
            else:
                tau_new = self.tau_init
            
            method = 'fallback_quantile'
            mu, sigma = None, None
        else:
            scores_tensor = torch.tensor(list(self.score_buffer))
            mu, sigma = self._fit_lognormal_mle(scores_tensor)
            tau_new = self._compute_threshold_from_lognormal(scores_tensor, mu, sigma)
            method = 'lognormal_mle'
        
        self.lognormal_params = {
            'mu': mu,
            'sigma': sigma,
            'n_samples': n_samples,
            'quantile': self.quantile,
        }
        
        tau_smoothed = (1 - self.ema_alpha) * self.threshold + self.ema_alpha * tau_new
        tau_smoothed = max(self.tau_min, min(self.tau_max, tau_smoothed))
        
        self.threshold = tau_smoothed
        self.update_count += 1
        self.last_update_info = {
            'tau': tau_smoothed,
            'tau_raw': tau_new,
            'n_samples': n_samples,
            'method': method,
            'mu': mu,
            'sigma': sigma,
        }
        
        return tau_smoothed
    
    def get_threshold_tensor(self, device: torch.device = None) -> float:
        """
        获取全局单一阈值（标量值）
        
        Returns:
            全局阈值（float）
        """
        return self.threshold
    
    def get_status_string(self) -> str:
        """获取状态字符串（用于日志）"""
        n = self.last_update_info.get('samples', len(self.score_buffer)) if self.last_update_info else len(self.score_buffer)
        mu = self.lognormal_params.get('mu', None) if self.lognormal_params else None
        sigma = self.lognormal_params.get('sigma', None) if self.lognormal_params else None
        
        if mu is not None and sigma is not None:
            return f'threshold:{self.threshold:.3f} | n:{n} | μ:{mu:.3f} | σ:{sigma:.3f}'
        else:
            return f'threshold:{self.threshold:.3f} | n:{n}'
    
    def state_dict(self) -> Dict:
        """保存状态"""
        return {
            'threshold': self.threshold,
            'update_count': self.update_count,
            'score_buffer': list(self.score_buffer),
            'lognormal_params': self.lognormal_params,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """加载状态"""
        self.threshold = state.get('threshold', self.threshold)
        self.update_count = state.get('update_count', 0)
        buf_data = state.get('score_buffer', [])
        self.score_buffer = deque(buf_data, maxlen=self.window_size)
        self.lognormal_params = state.get('lognormal_params', None)


# ============================================================================
# -1. 伪标签可视化辅助函数（训练时调用）
# ============================================================================

def visualize_pseudo_labels_during_training(
    images_weak: torch.Tensor,
    pseudo_boxes: torch.Tensor,
    pseudo_labels: torch.Tensor,
    pseudo_scores: torch.Tensor,
    save_path: str,
    class_names: Optional[List[str]] = None,
    title: str = "Filtered Pseudo Labels",
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> None:
    """
    训练时可视化筛选后的伪标签
    
    Args:
        images_weak: 弱增强图像 [B, 3, H, W] 或 dict{'rgb': ...}
        pseudo_boxes: 伪标签框 [N, 4]，格式 cxcywh 归一化坐标
        pseudo_labels: 伪标签类别 [N]
        pseudo_scores: 伪标签置信度 [N]
        save_path: 保存路径（如 'visualization/pseudo_label_iter_100.jpg'）
        class_names: 类别名称列表，如 ['small', 'big']
        title: 图像标题
        mean: 图像归一化均值
        std: 图像归一化标准差
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        from pathlib import Path
    except ImportError:
        print("⚠️ 可视化依赖 PIL，请安装：pip install pillow")
        return
    
    # 确保保存目录存在
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 提取 RGB 图像
    if isinstance(images_weak, dict):
        rgb = images_weak.get('rgb', images_weak.get('images', None))
        if rgb is None:
            print(f"⚠️ 可视化错误：字典格式的 images_weak 中没有找到 'rgb' 键")
            return
    else:
        rgb = images_weak
    
    # 确保是 tensor
    if not isinstance(rgb, torch.Tensor):
        print(f"⚠️ 可视化错误：images_weak 不是 torch.Tensor 类型")
        return
    
    # 只取第一张图（batch 中的第一张）
    if rgb.dim() == 4:
        rgb = rgb[0]
    elif rgb.dim() != 3:
        print(f"⚠️ 可视化错误：图像维度不正确，期望 3 或 4 维，得到 {rgb.dim()} 维")
        return
    
    # 反归一化
    rgb_np = rgb.detach().float().cpu().numpy()
    rgb_np = rgb_np.transpose(1, 2, 0)  # CHW -> HWC
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    rgb_np = rgb_np * std_arr + mean_arr
    rgb_np = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(rgb_np, mode="RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # 转换为 numpy（确保在 CPU 上）
    if isinstance(pseudo_boxes, torch.Tensor):
        pseudo_boxes = pseudo_boxes.detach().cpu().numpy()
    elif not isinstance(pseudo_boxes, np.ndarray):
        print(f"⚠️ 可视化错误：pseudo_boxes 类型不正确：{type(pseudo_boxes)}")
        return
    
    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = pseudo_labels.detach().cpu().numpy()
    elif not isinstance(pseudo_labels, np.ndarray):
        print(f"⚠️ 可视化错误：pseudo_labels 类型不正确：{type(pseudo_labels)}")
        return
    
    if isinstance(pseudo_scores, torch.Tensor):
        pseudo_scores = pseudo_scores.detach().cpu().numpy()
    elif not isinstance(pseudo_scores, np.ndarray):
        print(f"⚠️ 可视化错误：pseudo_scores 类型不正确：{type(pseudo_scores)}")
        return
    
    # 确保长度一致
    if len(pseudo_boxes) != len(pseudo_labels) or len(pseudo_boxes) != len(pseudo_scores):
        print(f"⚠️ 可视化错误：伪标签数量不一致 - boxes:{len(pseudo_boxes)}, labels:{len(pseudo_labels)}, scores:{len(pseudo_scores)}")
        return
    
    # 绘制标题
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([0, 0, w, 35], fill=(0, 0, 0))
    title_text = f"{title} | n={len(pseudo_boxes)}"
    draw.text((10, 8), title_text, fill=(255, 255, 255), font=font)
    
    # 绘制框
    for i in range(len(pseudo_boxes)):
        cx, cy, bw, bh = pseudo_boxes[i]
        
        if bw <= 0 or bh <= 0:
            continue
        
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        lab = int(pseudo_labels[i]) if i < len(pseudo_labels) else -1
        score = float(pseudo_scores[i]) if i < len(pseudo_scores) else 0.0
        
        # 配色：label=0 绿色，label=1 红色，其他橙色
        if lab == 0:
            color = (0, 255, 0)  # green
        elif lab == 1:
            color = (255, 0, 0)  # red
        else:
            color = (255, 165, 0)  # orange
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # 标签文本
        name = str(lab)
        if class_names is not None and 0 <= lab < len(class_names):
            name = class_names[lab]
        text = f"{name}:{score:.2f}"
        
        # 文本背景
        if font is not None:
            try:
                tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            except Exception:
                tw, th = len(text) * 6, 12
        else:
            tw, th = len(text) * 6, 12
        
        y_text = max(36, y1 - th - 4)
        label_x2 = min(w - 1, x1 + tw + 4)
        label_y2 = min(h - 1, y_text + th + 4)
        if x1 < label_x2 and y_text < label_y2:
            draw.rectangle([x1, y_text, label_x2, label_y2], fill=color)
        draw.text((x1 + 2, y_text + 2), text, fill=(255, 255, 255), font=font)
    
    # 保存
    img.save(str(save_path), quality=95)

# ============================================================================
# -0.1 从 encoder 特征提取 RoI embedding（用于语义原型计算）
# ============================================================================

def roi_embed_from_boxes(
    encoder_feats: List[torch.Tensor],
    boxes_cxcywh_norm: torch.Tensor,
    batch_ids: torch.Tensor,
    img_size: Tuple[int, int],
    featmap_strides: List[int] = [8, 16, 32],
    output_size: int = 7,
) -> torch.Tensor:
    """
    从多尺度 encoder 特征中提取指定框位置的 embedding
    
    Args:
        encoder_feats: 多尺度 encoder 特征 [[B, C, H1, W1], [B, C, H2, W2], ...]
        boxes_cxcywh_norm: 归一化坐标的框 [N, 4]，格式 cxcywh，范围 [0, 1]
        batch_ids: 每个框对应的 batch 索引 [N]
        img_size: 图像尺寸 (H, W)
        featmap_strides: 各层特征图的步长
        output_size: RoIAlign 输出尺寸
    
    Returns:
        embeddings: RoI embedding [N, C]（float32，未 L2 归一化）
    """
    from torchvision.ops import roi_align
    
    N = boxes_cxcywh_norm.shape[0]
    if N == 0 or encoder_feats is None or len(encoder_feats) == 0:
        C = encoder_feats[0].shape[1] if encoder_feats and len(encoder_feats) > 0 else 256
        device = boxes_cxcywh_norm.device if boxes_cxcywh_norm.numel() > 0 else torch.device('cpu')
        return torch.zeros(0, C, device=device, dtype=torch.float32)
    
    valid_boxes = torch.isfinite(boxes_cxcywh_norm).all(dim=1)
    if not valid_boxes.all():
        valid_indices = torch.where(valid_boxes)[0]
        if valid_indices.numel() == 0:
            C = encoder_feats[0].shape[1]
            return torch.zeros(0, C, device=boxes_cxcywh_norm.device, dtype=torch.float32)
        boxes_cxcywh_norm = boxes_cxcywh_norm[valid_indices]
        batch_ids = batch_ids[valid_indices]
        N = boxes_cxcywh_norm.shape[0]
    
    device = boxes_cxcywh_norm.device
    H_img, W_img = img_size
    C = encoder_feats[0].shape[1]
    num_levels = len(encoder_feats)
    
    cx, cy, w, h = boxes_cxcywh_norm[:, 0], boxes_cxcywh_norm[:, 1], boxes_cxcywh_norm[:, 2], boxes_cxcywh_norm[:, 3]
    x1 = (cx - w / 2) * W_img
    y1 = (cy - h / 2) * H_img
    x2 = (cx + w / 2) * W_img
    y2 = (cy + h / 2) * H_img
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
    
    box_w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    box_h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    roi_scales = torch.sqrt(box_w * box_h)
    finest_scale = 56.0
    target_lvls = torch.floor(torch.log2(roi_scales / finest_scale + 1e-6))
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
    
    feat_dtype = encoder_feats[0].dtype
    roi_feats_all = torch.zeros((N, C, output_size, output_size), device=device, dtype=feat_dtype)
    for level in range(num_levels):
        mask = (target_lvls == level)
        level_indices = torch.where(mask)[0]
        if level_indices.numel() == 0:
            continue
        
        level_boxes = boxes_xyxy[level_indices]
        level_batch_ids = batch_ids[level_indices]
        
        stride = featmap_strides[level]
        spatial_scale = 1.0 / stride
        
        rois = torch.cat([level_batch_ids.unsqueeze(-1).float(), level_boxes], dim=-1)
        rois = rois.to(dtype=feat_dtype)
        
        roi_feats = roi_align(
            encoder_feats[level],
            rois,
            output_size=(output_size, output_size),
            spatial_scale=spatial_scale,
            sampling_ratio=0,
            aligned=True
        )
        
        roi_feats_all[level_indices] = roi_feats
    
    embeddings = roi_feats_all.float().mean(dim=[2, 3])
    
    return embeddings


def extract_decoder_features_for_gt(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict],
    device: torch.device = None,
    use_last_layer: bool = True,
    min_iou_threshold: float = 0.3,  # 🆕 最小 IoU 阈值，低于此值的匹配将被过滤
) -> torch.Tensor:
    """
    从 decoder 输出中提取与 GT 框匹配的 query 特征
        
        Args:
        outputs: 模型输出字典，包含:
            - 'pred_boxes': [B, Q, 4] 预测框 (cxcywh, normalized)
            - 'dec_hs': dict {layer_idx: [B, Q, C]} decoder hidden states
        targets: GT 标注列表，每个元素包含 'boxes' [N, 4] (cxcywh, normalized)
        device: 设备
        use_last_layer: 是否只使用最后一层 decoder 特征（默认 True）
        min_iou_threshold: 最小 IoU 阈值（默认 0.3），低于此值的匹配将被过滤
            
        Returns:
        gt_decoder_feats: [total_valid_gt_boxes, C] 所有有效 GT 框对应的 decoder 特征
    """
    pred_boxes = outputs.get('pred_boxes')  # [B, Q, 4]
    dec_hs = outputs.get('dec_hs')  # dict {layer_idx: [B, Q, C]} 或 list
    
    if pred_boxes is None or dec_hs is None:
        return None
    
    if device is None:
        device = pred_boxes.device
    
    B, Q, _ = pred_boxes.shape
    
    # 获取 decoder 特征（使用最后一层）
    if isinstance(dec_hs, dict):
        # dict 格式：{0: tensor, 1: tensor, ...}
        layer_keys = sorted(dec_hs.keys())
        if use_last_layer:
            decoder_feats = dec_hs[layer_keys[-1]]  # [B, Q, C]
        else:
            # 平均所有层
            decoder_feats = torch.stack([dec_hs[k] for k in layer_keys], dim=0).mean(dim=0)
    elif isinstance(dec_hs, (list, tuple)):
        # list 格式
        if use_last_layer:
            decoder_feats = dec_hs[-1]  # [B, Q, C]
        else:
            decoder_feats = torch.stack(dec_hs, dim=0).mean(dim=0)
    else:
        return None
    
    C = decoder_feats.shape[-1]
    
    all_gt_feats = []
    total_gt_count = 0
    valid_gt_count = 0
    all_valid_ious = []
    
    for b in range(B):
        if b >= len(targets):
            continue
            
        gt_boxes = targets[b].get('boxes', None)  # [N, 4]
        if gt_boxes is None or gt_boxes.shape[0] == 0:
            continue
        
        N = gt_boxes.shape[0]
        total_gt_count += N
        pred_boxes_b = pred_boxes[b]  # [Q, 4]
        decoder_feats_b = decoder_feats[b]  # [Q, C]
        
        gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_b)
        
        iou_matrix, _ = box_iou(gt_boxes_xyxy, pred_boxes_xyxy)
        best_iou_values, best_pred_indices = iou_matrix.max(dim=1)
    
        valid_mask = best_iou_values >= min_iou_threshold
        
        if valid_mask.any():
            valid_indices = best_pred_indices[valid_mask]
            valid_gt_matched_feats = decoder_feats_b[valid_indices]
            all_gt_feats.append(valid_gt_matched_feats)
            
            num_valid = valid_mask.sum().item()
            valid_gt_count += num_valid
            all_valid_ious.append(best_iou_values[valid_mask].cpu())
    
    if len(all_gt_feats) == 0:
        if total_gt_count > 0:
            print(f"警告：所有 {total_gt_count} 个 GT 框的 IoU 都低于阈值 {min_iou_threshold}，无法提取 decoder 特征")
        return torch.zeros(0, C, device=device, dtype=decoder_feats.dtype)
    
    result = torch.cat(all_gt_feats, dim=0)
    
    if total_gt_count > 0:
        filter_rate = (total_gt_count - valid_gt_count) / total_gt_count
        if filter_rate > 0.3:
            all_valid_ious_tensor = torch.cat(all_valid_ious) if all_valid_ious else torch.tensor([])
            avg_iou = all_valid_ious_tensor.mean().item() if len(all_valid_ious_tensor) > 0 else 0.0
            print(f"Decoder 特征匹配：{valid_gt_count}/{total_gt_count} 个 GT 通过 IoU 阈值 {min_iou_threshold}，平均 IoU={avg_iou:.3f}")
    
    return result


# ============================================================================
# 0. 语义原型伪标签筛选模块（队列存储 + 多尺度 RoIAlign）
# ============================================================================

from collections import deque
from torchvision.ops import roi_align


class SemanticPrototypeQueue:
    """
    语义原型队列管理器
    
    - 每个类别维护一个长度为 50 的特征队列
    - 使用多尺度 RoIAlign 提取特征
    - 计算原型为队列特征的平均值
    """
    
    def __init__(self, queue_size: int = 50, num_classes: int = 2, feature_dim: int = 256):
        """
        Args:
            queue_size: 队列长度（默认 50）
            num_classes: 类别数量
            feature_dim: 特征维度
        """
        self.queue_size = queue_size
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 每个类别一个队列，存储特征
        self.queues = {i: deque(maxlen=queue_size) for i in range(num_classes)}
        
        # 缓存的原型（队列特征的平均值）
        self._cached_prototypes = {}
        self._cache_valid = {i: False for i in range(num_classes)}
        
        # 统计信息
        self.total_added = {i: 0 for i in range(num_classes)}
    
    @torch.no_grad()
    def extract_multi_scale_roi_features(
        self,
        encoder_feats: List[torch.Tensor],
        boxes: torch.Tensor,
        batch_idx: int,
        img_size: Tuple[int, int],
        featmap_strides: List[int] = [8, 16, 32],
        output_size: int = 7,
    ) -> torch.Tensor:
        """
        使用 MultiScaleRoIAlign 提取框特征（优化版，避免 GPU/CPU 同步）
        
        Args:
            encoder_feats: 多尺度 encoder 特征列表 [[B, C, H1, W1], ...]
            boxes: 框坐标 [N, 4] (cxcywh, normalized)
            batch_idx: batch 索引
            img_size: 图像尺寸 (H, W)
            featmap_strides: 各层特征图的步长
            output_size: RoIAlign 输出尺寸
        
        Returns:
            features: [N, C] 提取的特征（L2 归一化）
        """
        from torchvision.ops import MultiScaleRoIAlign
        
        N = len(boxes)
        if N == 0:
            return torch.zeros(0, self.feature_dim, device=boxes.device)
        
        device = boxes.device
        H_img, W_img = img_size
        C = encoder_feats[0].shape[1]
        feat_dtype = encoder_feats[0].dtype
        
        # 清理 NaN/Inf 并裁剪到有效范围
        boxes = torch.nan_to_num(boxes, nan=0.5, posinf=0.99, neginf=0.01)
        boxes = boxes.clamp(0.01, 0.99)
        
        # 转换 cxcywh -> xyxy (绝对坐标)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2) * W_img
        y1 = (cy - h / 2) * H_img
        x2 = (cx + w / 2) * W_img
        y2 = (cy + h / 2) * H_img
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # 确保框有效（最小尺寸 1 像素）
        boxes_xyxy = boxes_xyxy.clamp(min=0.0)
        boxes_xyxy[:, 2] = boxes_xyxy[:, 2].clamp(max=W_img)
        boxes_xyxy[:, 3] = boxes_xyxy[:, 3].clamp(max=H_img)
        # 确保 x2 > x1, y2 > y1
        boxes_xyxy[:, 2] = torch.max(boxes_xyxy[:, 2], boxes_xyxy[:, 0] + 1)
        boxes_xyxy[:, 3] = torch.max(boxes_xyxy[:, 3], boxes_xyxy[:, 1] + 1)
        
        # 构建特征图字典（MultiScaleRoIAlign 需要的格式）
        featmap_names = [str(i) for i in range(len(encoder_feats))]
        features_dict = {name: feat for name, feat in zip(featmap_names, encoder_feats)}
        
        # 创建 MultiScaleRoIAlign（使用 FPN 层级映射）
        ms_roi_align = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=output_size,
            sampling_ratio=0,
        )
        
        # 将 boxes 转为列表格式（每张图一个 tensor）
        # 由于我们只处理单张图，创建一个包含所有框的列表
        boxes_list = [boxes_xyxy.to(dtype=feat_dtype)]
        
        # 构建图像尺寸列表
        image_shapes = [(H_img, W_img)]
        
        # 执行 MultiScaleRoIAlign
        try:
            roi_feats = ms_roi_align(features_dict, boxes_list, image_shapes)  # [N, C, output_size, output_size]
        except Exception:
            # 如果 MultiScaleRoIAlign 失败，回退到简单的单尺度 roi_align
            batch_ids = torch.zeros(N, 1, dtype=feat_dtype, device=device)
            rois = torch.cat([batch_ids, boxes_xyxy.to(dtype=feat_dtype)], dim=-1)
            roi_feats = roi_align(
                encoder_feats[0],  # 使用最高分辨率特征
                rois,
                output_size=(output_size, output_size),
                spatial_scale=1.0 / featmap_strides[0],
                sampling_ratio=0,
                aligned=True,
            )
        
        # 全局平均池化 + L2 归一化
        features = roi_feats.mean(dim=[2, 3])
        features = F.normalize(features.float(), p=2, dim=-1)
        
        return features
    
    @torch.no_grad()
    def add_features(
        self,
        encoder_feats: List[torch.Tensor],
        boxes: torch.Tensor,
        labels: torch.Tensor,
        batch_idx: int,
        img_size: Tuple[int, int],
    ):
        """
        添加标注数据的特征到队列
        
        Args:
            encoder_feats: 多尺度 encoder 特征列表
            boxes: GT 框 [N, 4] (cxcywh, normalized)
            labels: GT 标签 [N]
            batch_idx: batch 索引
            img_size: 图像尺寸 (H, W)
        """
        if len(boxes) == 0:
            return
        
        features = self.extract_multi_scale_roi_features(
            encoder_feats, boxes, batch_idx, img_size
        )
        
        labels_cpu = labels.cpu().tolist()
        for i, label_id in enumerate(labels_cpu):
            if 0 <= label_id < self.num_classes:
                self.queues[label_id].append(features[i].clone())
                self._cache_valid[label_id] = False
                self.total_added[label_id] += 1
    
    def get_prototype(self, class_id: int) -> Optional[torch.Tensor]:
        """
        获取指定类别的原型（队列特征的平均值）
        
        Args:
            class_id: 类别 ID
        
        Returns:
            prototype: [C] 原型特征，如果队列为空则返回 None
        """
        if class_id not in self.queues or len(self.queues[class_id]) == 0:
            return None
        
        if self._cache_valid.get(class_id, False) and class_id in self._cached_prototypes:
            return self._cached_prototypes[class_id]
        
        # 统一设备：优先使用 CUDA 设备（新添加的特征通常在 GPU 上）
        queue_list = list(self.queues[class_id])
        
        # 查找第一个 CUDA 设备，如果没有就使用第一个特征的设备
        target_device = queue_list[0].device
        for feat in queue_list:
            if feat.device.type == 'cuda':
                target_device = feat.device
                break
        
        # 将所有特征移到目标设备
        queue_list = [feat.to(target_device) for feat in queue_list]
        
        queue_features = torch.stack(queue_list, dim=0)
        prototype = queue_features.mean(dim=0)
        prototype = F.normalize(prototype, p=2, dim=-1)
        
        self._cached_prototypes[class_id] = prototype
        self._cache_valid[class_id] = True
        
        return prototype
    
    def get_all_prototypes(self) -> Dict[int, torch.Tensor]:
        """获取所有类别的原型"""
        prototypes = {}
        for class_id in range(self.num_classes):
            proto = self.get_prototype(class_id)
            if proto is not None:
                prototypes[class_id] = proto
        return prototypes
    
    def get_queue_features(self, class_id: int) -> Optional[torch.Tensor]:
        """
        获取指定类别队列中的所有特征（用于逐一比较）
        
        Args:
            class_id: 类别 ID
        
        Returns:
            features: [Q, C] 队列中的所有特征，如果队列为空则返回 None
        """
        if class_id not in self.queues or len(self.queues[class_id]) == 0:
            return None
        
        # 统一设备：优先使用 CUDA 设备（新添加的特征通常在 GPU 上）
        queue_list = list(self.queues[class_id])
        
        # 查找第一个 CUDA 设备，如果没有就使用第一个特征的设备
        target_device = queue_list[0].device
        for feat in queue_list:
            if feat.device.type == 'cuda':
                target_device = feat.device
                break
        
        # 将所有特征移到目标设备
        queue_list = [feat.to(target_device) for feat in queue_list]
        
        return torch.stack(queue_list, dim=0)
    
    def get_queue_length(self, class_id: int) -> int:
        """获取指定类别队列的当前长度"""
        if class_id not in self.queues:
            return 0
        return len(self.queues[class_id])
    
    def state_dict(self) -> Dict:
        """获取状态字典（用于保存 checkpoint）"""
        state = {
            'queue_size': self.queue_size,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'queues': {},
            'total_added': self.total_added,
        }
        
        for class_id, queue in self.queues.items():
            if len(queue) > 0:
                # 统一设备后再保存（移到 CPU 以节省显存）
                queue_list = list(queue)
                target_device = queue_list[0].device
                queue_list = [feat.to(target_device) for feat in queue_list]
                state['queues'][class_id] = torch.stack(queue_list, dim=0).cpu()
        
        return state
    
    def load_state_dict(self, state: Dict):
        """加载状态字典（用于恢复 checkpoint）"""
        self.queue_size = state.get('queue_size', self.queue_size)
        self.num_classes = state.get('num_classes', self.num_classes)
        self.feature_dim = state.get('feature_dim', self.feature_dim)
        self.total_added = state.get('total_added', {i: 0 for i in range(self.num_classes)})
        
        self.queues = {i: deque(maxlen=self.queue_size) for i in range(self.num_classes)}
        for class_id, features in state.get('queues', {}).items():
            class_id = int(class_id)
            if class_id < self.num_classes:
                # 确保所有加载的特征都在 CPU 上（后续使用时会自动移到正确设备）
                for feat in features:
                    self.queues[class_id].append(feat.cpu())
        
        self._cache_valid = {i: False for i in range(self.num_classes)}
    
    def get_statistics(self) -> Dict:
        """获取队列统计信息"""
        return {
            'queue_size': self.queue_size,
            'num_classes': self.num_classes,
            'queue_lengths': {i: len(q) for i, q in self.queues.items()},
            'total_added': self.total_added,
        }
    
    def __repr__(self):
        lengths = [len(self.queues[i]) for i in range(self.num_classes)]
        return f"SemanticPrototypeQueue(queue_size={self.queue_size}, classes={self.num_classes}, lengths={lengths})"


_GLOBAL_PROTOTYPE_QUEUE: Optional[SemanticPrototypeQueue] = None


def get_prototype_queue(num_classes: int = 2, feature_dim: int = 256, queue_size: int = 50) -> SemanticPrototypeQueue:
    """获取或创建全局原型队列"""
    global _GLOBAL_PROTOTYPE_QUEUE
    if _GLOBAL_PROTOTYPE_QUEUE is None:
        _GLOBAL_PROTOTYPE_QUEUE = SemanticPrototypeQueue(
            queue_size=queue_size,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
    return _GLOBAL_PROTOTYPE_QUEUE


def reset_prototype_queue():
    """重置全局原型队列"""
    global _GLOBAL_PROTOTYPE_QUEUE
    _GLOBAL_PROTOTYPE_QUEUE = None


@torch.no_grad()
def update_global_prototypes(
    encoder_feats: List[torch.Tensor],
    boxes: torch.Tensor,
    labels: torch.Tensor,
    batch_idx: int,
    img_size: Tuple[int, int],
    num_classes: int = 2,
    queue_size: int = 50,
):
    """
    使用监督数据的 GT 框更新全局类别原型队列（多尺度 RoIAlign 方式）
    """
    if len(boxes) == 0 or encoder_feats is None or len(encoder_feats) == 0:
        return
    
    C = encoder_feats[0].shape[1]
    proto_queue = get_prototype_queue(num_classes=num_classes, feature_dim=C, queue_size=queue_size)
    proto_queue.add_features(encoder_feats, boxes, labels, batch_idx, img_size)


@torch.no_grad()
def _filter_by_prototype_similarity(
    encoder_feats: List[torch.Tensor],
    boxes: torch.Tensor,
    labels: torch.Tensor,
    batch_idx: int,
    threshold: float = 0.9,  # 🔥 固定阈值 0.9
    img_size: Tuple[int, int] = None,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    基于语义原型相似度筛选伪标签（改进版：与原型比较，而非与每个队列特征比较）
    
    核心思想：
    1. 计算每个类别的原型（队列特征的均值）
    2. 计算伪标签特征与同类原型的余弦相似度
    3. 使用固定阈值 0.9 筛选：相似度 >= 0.9 的伪标签被保留
    
    Args:
        encoder_feats: 多尺度 encoder 特征列表
        boxes: 伪标签框 [N, 4]
        labels: 伪标签类别 [N]
        batch_idx: batch 索引
        threshold: 固定相似度阈值（默认 0.9，实际代码中硬编码为 0.9）
        img_size: 图像尺寸
        num_classes: 类别数量
    
    Returns:
        keep_mask: 布尔掩码 [N]
    """
    N = len(boxes)
    if N == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes.device)
    
    device = boxes.device
    
    if encoder_feats is None or len(encoder_feats) == 0:
        return torch.ones(N, dtype=torch.bool, device=device)
    
    C = encoder_feats[0].shape[1]
    proto_queue = get_prototype_queue(num_classes=num_classes, feature_dim=C)
    
    # 提取伪标签的特征
    pseudo_features = proto_queue.extract_multi_scale_roi_features(
        encoder_feats, boxes, batch_idx, img_size
    )
    pseudo_features = pseudo_features.to(device)
    
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)
    all_sims = torch.full((N,), -1.0, device=device)  # 🔥 记录所有伪标签的相似度得分
    
    for class_id in range(num_classes):
        class_mask = (labels == class_id)
        class_indices = torch.where(class_mask)[0]
        if class_indices.numel() == 0:
            continue
        
        # 🔥 获取类别原型（队列特征的均值，已 L2 归一化）
        prototype = proto_queue.get_prototype(class_id)
        if prototype is None:
            # 队列为空时，保留所有该类别的伪标签
            all_sims[class_indices] = 1.0  # 设为最高相似度
            continue
        
        prototype = prototype.to(device)
        class_feats = pseudo_features[class_indices]  # [M, C]
        
        # 🔥 计算伪标签特征与原型的余弦相似度
        # class_feats 已经 L2 归一化，prototype 也已 L2 归一化
        pseudo_to_proto_sim = torch.mv(class_feats, prototype)  # [M]
        pseudo_to_proto_sim = torch.nan_to_num(pseudo_to_proto_sim, nan=0.0, posinf=1.0, neginf=0.0)
        all_sims[class_indices] = pseudo_to_proto_sim
        
        # 🔥 使用固定阈值 0.8 筛选
        fixed_threshold = 0.8
        class_keep = pseudo_to_proto_sim >= fixed_threshold
        
        keep_mask[class_indices] = class_keep
    
    return keep_mask


def get_global_prototypes():
    """获取当前全局原型（用于调试和保存）"""
    global _GLOBAL_PROTOTYPE_QUEUE
    if _GLOBAL_PROTOTYPE_QUEUE is None:
        return {}, {}
    return _GLOBAL_PROTOTYPE_QUEUE.get_all_prototypes(), _GLOBAL_PROTOTYPE_QUEUE.get_statistics()


def get_prototype_state_dict():
    """获取原型队列状态字典（用于保存 checkpoint）"""
    global _GLOBAL_PROTOTYPE_QUEUE
    if _GLOBAL_PROTOTYPE_QUEUE is None:
        return None
    return _GLOBAL_PROTOTYPE_QUEUE.state_dict()


def load_prototype_state_dict(state: Dict):
    """加载原型队列状态字典（用于恢复 checkpoint）"""
    global _GLOBAL_PROTOTYPE_QUEUE
    if state is None:
        return
    
    num_classes = state.get('num_classes', 2)
    feature_dim = state.get('feature_dim', 256)
    queue_size = state.get('queue_size', 50)
    
    _GLOBAL_PROTOTYPE_QUEUE = SemanticPrototypeQueue(
        queue_size=queue_size,
        num_classes=num_classes,
        feature_dim=feature_dim,
    )
    _GLOBAL_PROTOTYPE_QUEUE.load_state_dict(state)


# ============================================================================
# 0.1 VDVI 植被指数筛选模块
# ============================================================================

def filter_pseudo_labels_by_vdvi(
    boxes: torch.Tensor,
    vdvi_map: torch.Tensor,
    vdvi_thresh: float = 0.030,
    box_format: str = 'cxcywh',
    center_patch_size: int = 5,  # 保留参数以保持兼容性，但不使用
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据 VDVI（可见光差分植被指数）筛选伪标签
    
    VDVI = (2*G - R - B) / (2*G + R + B)，值域 [-1, 1]
    
    Args:
        boxes: 目标框张量，形状 [N, 4]
            - 如果 box_format='cxcywh'，格式为 (cx, cy, w, h)，值域 [0, 1] 归一化
            - 如果 box_format='xyxy'，格式为 (x1, y1, x2, y2)，值域 [0, 1] 归一化
        vdvi_map: VDVI 图，形状 [H, W]，值域 [-1, 1]
        vdvi_thresh: VDVI 阈值，大于此值的区域被认为是植被（默认 0.030）
        box_format: 边界框格式，'cxcywh' 或 'xyxy'
        center_patch_size: [已弃用] 保留以保持兼容性
        
    Returns:
        keep_mask: 布尔掩码，形状 [N]，True 表示保留该伪标签
        vdvi_scores: 每个框的 VDVI 分数 [N]（用于统计）
    """
    if boxes.shape[0] == 0:
        return torch.ones(0, dtype=torch.bool, device=boxes.device), torch.zeros(0, device=boxes.device)
    
    H, W = vdvi_map.shape
    device = boxes.device
    N = boxes.shape[0]
    
    vdvi_map = vdvi_map.to(device)
    
    if box_format == 'cxcywh':
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1_norm = cx - w / 2
        y1_norm = cy - h / 2
        x2_norm = cx + w / 2
        y2_norm = cy + h / 2
    else:
        x1_norm, y1_norm, x2_norm, y2_norm = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    grid_size = 15
    
    grid_coords = torch.linspace(0, 1, grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(grid_coords, grid_coords, indexing='ij')
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)
    
    x1_pixel = (x1_norm * W).clamp(0, W - 1)
    y1_pixel = (y1_norm * H).clamp(0, H - 1)
    x2_pixel = (x2_norm * W).clamp(0, W - 1)
    y2_pixel = (y2_norm * H).clamp(0, H - 1)
    
    x2_pixel = torch.max(x2_pixel, x1_pixel + 1)
    y2_pixel = torch.max(y2_pixel, y1_pixel + 1)
    
    box_widths = x2_pixel - x1_pixel
    box_heights = y2_pixel - y1_pixel
    
    sample_x = x1_pixel.unsqueeze(1) + grid_x.unsqueeze(0) * box_widths.unsqueeze(1)
    sample_y = y1_pixel.unsqueeze(1) + grid_y.unsqueeze(0) * box_heights.unsqueeze(1)
    
    sample_x = sample_x.long().clamp(0, W - 1)
    sample_y = sample_y.long().clamp(0, H - 1)
    
    sampled_values = vdvi_map[sample_y, sample_x]
    
    vdvi_scores = sampled_values.mean(dim=1)
    keep_mask = vdvi_scores > vdvi_thresh
    
    return keep_mask, vdvi_scores


# ============================================================================
# 0.1. 跨类别 NMS（优化版：使用 torchvision.ops.nms）
# ============================================================================

def cross_class_nms(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,  # 🔥 新增：包含度阈值
) -> torch.Tensor:
    """
    改进版跨类别 NMS：同时处理重叠和嵌套检测
    
    Args:
        boxes: 框坐标 [N, 4]，格式 cxcywh（归一化坐标）
        labels: 类别标签 [N]（此函数忽略类别）
        scores: 置信度分数 [N]
        iou_threshold: NMS IoU 阈值，默认 0.5
        containment_threshold: 包含度阈值，默认 0.8（小框被大框包含 80% 以上则抑制）
        
    Returns:
        keep: 保留的框索引的布尔掩码 [N]
    """
    from torchvision.ops import nms
    
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes.device)
    
    if boxes.shape[0] == 1:
        return torch.ones(1, dtype=torch.bool, device=boxes.device)
    
    device = boxes.device
    N = boxes.shape[0]
    
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    
    # 使用 nan_to_num 处理潜在的 NaN/Inf，避免 GPU/CPU 同步
    boxes_xyxy = torch.nan_to_num(boxes_xyxy, nan=0.0, posinf=1.0, neginf=0.0)
    scores_clean = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    keep_indices_local = nms(boxes_xyxy, scores_clean, iou_threshold)
    keep_indices = keep_indices_local
    
    keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
    keep_mask[keep_indices] = True
    
    num_kept = keep_indices.shape[0]
    if num_kept > 1:
        kept_boxes = boxes_xyxy[keep_indices]
        kept_areas = (kept_boxes[:, 2] - kept_boxes[:, 0]) * (kept_boxes[:, 3] - kept_boxes[:, 1])
        
        xx1 = torch.max(kept_boxes[:, None, 0], kept_boxes[None, :, 0])
        yy1 = torch.max(kept_boxes[:, None, 1], kept_boxes[None, :, 1])
        xx2 = torch.min(kept_boxes[:, None, 2], kept_boxes[None, :, 2])
        yy2 = torch.min(kept_boxes[:, None, 3], kept_boxes[None, :, 3])
        
        inter_w = (xx2 - xx1).clamp(min=0)
        inter_h = (yy2 - yy1).clamp(min=0)
        inter_areas = inter_w * inter_h
        
        smaller_areas = torch.min(kept_areas[:, None], kept_areas[None, :])
        containment_matrix = inter_areas / (smaller_areas + 1e-8)
        
        upper_tri = torch.triu(torch.ones(num_kept, num_kept, device=device, dtype=torch.bool), diagonal=1)
        high_containment = (containment_matrix >= containment_threshold) & upper_tri
        
        suppressed_mask = high_containment.any(dim=0)
        keep_mask[keep_indices[suppressed_mask]] = False
    
    return keep_mask


# ============================================================================
# 0.5. One-to-Many 匹配器（参考 Semi-DETR）
# ============================================================================

class O2MAssigner:
    """
    One-to-Many 匹配器（参考 Semi-DETR 的 O2MAssigner）
    """
    
    def __init__(self, candidate_topk: int = 13):
        """
        Args:
            candidate_topk: 每个 GT 选择的候选 query 数量
        """
        self.candidate_topk = candidate_topk
    
    def assign(
        self,
        bbox_pred: torch.Tensor,
        cls_pred: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        img_size: Tuple[int, int],
        alpha: float = 1.0,
        beta: float = 6.0,
    ) -> Dict:
        """
        执行 one-to-many 匹配
        
        Args:
            bbox_pred: 预测框，形状 [num_query, 4]，格式 (cx, cy, w, h)，归一化坐标 [0, 1]
            cls_pred: 预测分类分数，形状 [num_query, num_classes]，sigmoid 后的概率
            gt_bboxes: GT 框，形状 [num_gt, 4]，格式 (cx, cy, w, h)，归一化坐标 [0, 1]
            gt_labels: GT 标签，形状 [num_gt]
            img_size: 图像尺寸 (H, W)
            alpha: 分类分数权重（对齐分数公式中的 α）
            beta: IoU 权重（对齐分数公式中的 β）
            
        Returns:
            dict: 包含以下键：
                - assigned_gt_inds: [num_query]，分配的 GT 索引（0=背景，1-based=正样本）
                - assigned_labels: [num_query]，分配的标签（-1=未分配）
                - max_overlaps: [num_query]，最大 IoU
                - assign_metrics: [num_query]，对齐分数
        """
        INF = 100000000
        num_gts, num_queries = gt_bboxes.size(0), bbox_pred.size(0)
        device = bbox_pred.device
        
        # 初始化分配结果
        assigned_gt_inds = torch.zeros(num_queries, dtype=torch.long, device=device)
        assigned_labels = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        assign_metrics = torch.zeros(num_queries, device=device)
        max_overlaps = torch.zeros(num_queries, device=device)
        
        if num_gts == 0 or num_queries == 0:
            return {
                'assigned_gt_inds': assigned_gt_inds,
                'assigned_labels': assigned_labels,
                'max_overlaps': max_overlaps,
                'assign_metrics': assign_metrics,
            }
        
        H_img, W_img = img_size
        factor = torch.tensor([W_img, H_img, W_img, H_img], device=device, dtype=bbox_pred.dtype).unsqueeze(0)
        
        # 转换预测框到绝对坐标 xyxy
        pred_bboxes_abs = box_cxcywh_to_xyxy(bbox_pred) * factor  # [num_query, 4]
        gt_bboxes_abs = box_cxcywh_to_xyxy(gt_bboxes) * factor    # [num_gt, 4]
        
        # 计算 IoU（box_iou 返回 (iou, union) 元组）
        overlaps, _ = box_iou(pred_bboxes_abs, gt_bboxes_abs)
        overlaps = overlaps.detach()  # [num_query, num_gt]
        
        # 获取每个 query 对对应 GT 类别的分类分数
        bbox_scores = cls_pred[:, gt_labels].detach()  # [num_query, num_gt]
        
        # 计算对齐分数：alignment_metric = (cls_score^α) × (IoU^β)
        alignment_metrics = (bbox_scores ** alpha) * (overlaps ** beta)  # [num_query, num_gt]
        
        # 为每个 GT 选择 top-k 候选 query
        _, candidate_idxs = alignment_metrics.topk(
            min(self.candidate_topk, num_queries), dim=0, largest=True
        )  # [topk, num_gt]
        
        candidate_metrics = alignment_metrics[candidate_idxs, torch.arange(num_gts, device=device)]
        
        # 所有候选都作为正样本（is_pos = candidate_metrics > 0）
        is_pos = candidate_metrics > 0
        
        # 处理索引：将候选索引映射到全局索引（向量化，避免 for 循环）
        gt_offsets = torch.arange(num_gts, device=device) * num_queries  # [num_gts]
        candidate_idxs = candidate_idxs + gt_offsets.unsqueeze(0)  # [topk, num_gts] 广播
        candidate_idxs = candidate_idxs.view(-1)
        
        # 处理一个 query 被分配给多个 GT 的情况：选择 IoU 最大的 GT
        # 参考 Semi-DETR 的实现：先将 overlaps 转置并展平，然后根据候选索引赋值
        overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)  # [num_gt * num_queries]
        index = candidate_idxs.view(-1)[is_pos.view(-1)]  # 获取有效的候选索引
        if len(index) > 0:
            overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gts, -1).t()  # [num_gts, num_queries] -> [num_queries, num_gts]
        
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        
        # 分配 GT 索引（0=背景，1-based=正样本）
        assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1
        
        # 分配对齐分数
        pos_mask = max_overlaps != -INF
        # 🔥 优化：使用 nonzero 替代 any()，避免 GPU/CPU 同步
        pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
        if pos_indices.numel() > 0:
            assign_metrics[pos_mask] = alignment_metrics[pos_mask, argmax_overlaps[pos_mask]]
        
        # 分配标签
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1)
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        
        return {
            'assigned_gt_inds': assigned_gt_inds,
            'assigned_labels': assigned_labels,
            'max_overlaps': max_overlaps,
            'assign_metrics': assign_metrics,
        }


def _compute_loss_with_o2m_matching(
    outputs: Dict,
    targets: List[Dict],
    criterion: nn.Module,
    img_size: Tuple[int, int],
    hybrid_topk: int = 13,
    alpha: float = 1.0,
    beta: float = 6.0,
) -> Dict[str, torch.Tensor]:
    """
    使用 one-to-many 匹配计算损失（warm-up 阶段）
    
    🔥 改进策略：基于匈牙利匹配扩展为一对多
    1. 先用 criterion.matcher 进行匈牙利匹配（保证每个 GT 至少有一个匹配）
    2. 再基于 L1 距离为每个 GT 选择额外的 top-k 候选
    3. 使用距离倒数作为权重（距离越近权重越大）
    
    Args:
        outputs: 模型输出
        targets: 目标列表
        criterion: 损失计算器（用于获取 matcher 和损失函数）
        img_size: 图像尺寸 (H, W)
        hybrid_topk: 每个 GT 分配的额外 query 数量
        alpha: 分类分数权重（暂未使用）
        beta: IoU 权重（暂未使用）
        
    Returns:
        loss_dict: 损失字典
    """
    from ..deim.box_ops import generalized_box_iou
    
    pred_logits = outputs['pred_logits']  # [B, Q, C]
    pred_boxes = outputs['pred_boxes']     # [B, Q, 4]
    B, Q, C = pred_logits.shape
    device = pred_logits.device
    
    loss_dict = {}
    
    # 🔥 Step 1: 使用匈牙利匹配获取基础的一对一匹配
    outputs_for_match = {k: v for k, v in outputs.items() if 'aux' not in k and k != 'cross_outputs'}
    hungarian_indices = criterion.matcher(outputs_for_match, targets)['indices']
    
    # 对每个样本扩展为一对多匹配
    all_indices = []
    all_weights = []
    
    for b in range(B):
        target = targets[b]
        gt_boxes = target['boxes']  # [N, 4] (cxcywh, normalized)
        gt_labels = target['labels']  # [N]
        num_gts = len(gt_boxes)
        
        if num_gts == 0:
            all_indices.append((torch.tensor([], dtype=torch.long, device=device),
                               torch.tensor([], dtype=torch.long, device=device)))
            all_weights.append(torch.tensor([], device=device))
            continue
        
        # 获取匈牙利匹配结果
        h_src_idx, h_tgt_idx = hungarian_indices[b]  # 一对一匹配
        
        # 🔥 Step 2: 计算所有 query 到所有 GT 的 L1 距离
        pred_boxes_b = pred_boxes[b]  # [Q, 4]
        # L1 距离矩阵 [Q, N]
        l1_dist = torch.cdist(pred_boxes_b, gt_boxes, p=1)  # [Q, N]
        
        # 🔥 Step 3: 为每个 GT 选择 top-k 最近的 query（包括匈牙利匹配的）
        # 选择距离最小的 top-k
        k = min(hybrid_topk, Q)
        topk_dist, topk_indices = l1_dist.topk(k, dim=0, largest=False)  # [k, N]
        
        # 🔥 完全向量化操作（避免 Python 循环导致 CPU-GPU 同步卡住训练）
        # topk_indices: [k, N], topk_dist: [k, N]
        
        # Step 1: 展平 top-k 结果为 (query_idx, gt_idx, distance) 三元组
        # src_indices: [k * N], tgt_indices: [k * N]
        src_indices = topk_indices.t().reshape(-1)  # [N, k] -> [N*k]
        tgt_indices = torch.arange(num_gts, device=device).unsqueeze(1).expand(-1, k).reshape(-1)  # [N*k]
        distances_flat = topk_dist.t().reshape(-1)  # [N*k]
        
        # Step 2: 计算权重（per-GT softmax）
        # 先 reshape 回 [N, k]，计算 softmax，再展平
        distances_per_gt = topk_dist.t()  # [N, k]
        weights_per_gt = F.softmax(-distances_per_gt.detach() * 10, dim=1)  # [N, k]
        weights = weights_per_gt.reshape(-1)  # [N*k]
        
        # Step 3: 简化版 - 直接添加匈牙利匹配对，不检查是否存在
        # 🔥 原因：检查操作（isin, any）会触发 GPU/CPU 同步，导致卡死
        # 简化策略：直接添加所有匈牙利匹配对，让后续的去重逻辑处理
        if h_src_idx.numel() > 0 and h_tgt_idx.numel() > 0:
            # 确保匈牙利匹配结果在正确设备上
            h_src_idx = h_src_idx.to(device)
            h_tgt_idx = h_tgt_idx.to(device)
            
            # 直接添加匈牙利匹配对，权重设为最高
            # 🔥 注意：weights.max() 会触发一次同步，但这是必要的（确保匈牙利匹配权重最高）
            max_weight = weights.max().detach()  # detach 避免梯度传播
            h_weights = torch.full((h_src_idx.shape[0],), max_weight * 2.0, device=device)
            
            src_indices = torch.cat([src_indices, h_src_idx])
            tgt_indices = torch.cat([tgt_indices, h_tgt_idx])
            weights = torch.cat([weights, h_weights])
        
        # 🔥 处理重复：一个 query 可能被分配给多个 GT，保留权重最大的
        # 完全向量化操作（避免 Python 循环导致 CPU-GPU 同步卡住训练）
        if len(src_indices) > 0:
            # 按 (src_indices, -weights) 排序，这样每个 query 的最大权重排在最前
            # 使用负权重使得 argsort 后最大权重在前
            sort_keys = src_indices.float() * 1e10 - weights  # 先按 query 排序，再按权重降序
            sort_idx = torch.argsort(sort_keys)
            sorted_src = src_indices[sort_idx]
            sorted_tgt = tgt_indices[sort_idx]
            sorted_weights = weights[sort_idx]
            
            # 找到每个 unique query 的第一次出现（即最大权重的那个）
            # diff != 0 的位置是新 query 的开始
            diff = torch.diff(sorted_src, prepend=sorted_src.new_tensor([-1]))
            first_occurrence_mask = diff != 0
            
            # 提取结果
            final_src = sorted_src[first_occurrence_mask]
            final_tgt = sorted_tgt[first_occurrence_mask]
            final_weights = sorted_weights[first_occurrence_mask]
            
            all_indices.append((final_src, final_tgt))
            all_weights.append(final_weights)
        else:
            all_indices.append((torch.tensor([], dtype=torch.long, device=device),
                               torch.tensor([], dtype=torch.long, device=device)))
            all_weights.append(torch.tensor([], device=device))
    
    # 🔥 计算损失（使用基于距离的权重）
    device = pred_logits.device
    cls_losses = []
    bbox_losses = []
    giou_losses = []
    total_boxes = 0
    
    for b in range(B):
        indices = all_indices[b]
        match_weights = all_weights[b]  # 🔥 使用新的 all_weights
        
        if len(indices[0]) == 0:
            continue
        
        src_idx, tgt_idx = indices
        target_dict = targets[b]
        
        # 分类损失
        target_labels = target_dict['labels'][tgt_idx]  # [M]
        
        target_classes = torch.full((Q,), criterion.num_classes, dtype=torch.long, device=device)
        target_classes[src_idx] = target_labels
        target_onehot = F.one_hot(target_classes, num_classes=criterion.num_classes + 1)[..., :-1]
        
        # 计算 IoU 作为软目标分数
        src_boxes = pred_boxes[b][src_idx]
        tgt_boxes = target_dict['boxes'][tgt_idx]
        H_img, W_img = img_size
        factor = torch.tensor([W_img, H_img, W_img, H_img], device=device, dtype=pred_boxes.dtype).unsqueeze(0)
        src_boxes_abs = box_cxcywh_to_xyxy(src_boxes) * factor
        tgt_boxes_abs = box_cxcywh_to_xyxy(tgt_boxes) * factor
        ious, _ = box_iou(src_boxes_abs, tgt_boxes_abs)
        ious = torch.diag(ious).detach().to(dtype=pred_logits.dtype)
        
        # 🔥 使用匹配权重作为软目标分数（而非 IoU）
        # 这确保了即使 IoU=0，也有非零的目标分数
        target_score_o = torch.zeros((Q,), dtype=pred_logits.dtype, device=device)
        # 🔥 修复：确保 match_weights 与 target_score_o 类型匹配
        target_score_o[src_idx] = match_weights.to(dtype=pred_logits.dtype)
        target_score = target_score_o.unsqueeze(-1) * target_onehot
        
        pred_score = F.sigmoid(pred_logits[b]).detach()
        if criterion.mal_alpha is not None:
            weight = criterion.mal_alpha * pred_score.pow(criterion.gamma) * (1 - target_onehot) + target_onehot
        else:
            weight = pred_score.pow(criterion.gamma) * (1 - target_onehot) + target_onehot
        
        cls_loss = F.binary_cross_entropy_with_logits(
            pred_logits[b], target_score, weight=weight, reduction='none'
        )
        cls_losses.append(cls_loss.mean(1).sum())
        
        # 回归损失（使用匹配权重加权）
        # 🔥 修复：确保 match_weights 与损失类型匹配
        match_weights_typed = match_weights.to(dtype=src_boxes.dtype)
        bbox_loss = F.l1_loss(src_boxes, tgt_boxes, reduction='none')  # [M, 4]
        bbox_loss = (bbox_loss * match_weights_typed.unsqueeze(-1)).sum() / max(match_weights_typed.sum(), 1)
        bbox_losses.append(bbox_loss)
        
        giou_loss = 1 - torch.diag(generalized_box_iou(src_boxes_abs, tgt_boxes_abs))
        giou_loss = (giou_loss * match_weights_typed).sum() / max(match_weights_typed.sum(), 1)
        giou_losses.append(giou_loss)
        
        total_boxes += len(src_idx)
    
    # 归一化
    device = pred_logits.device
    Q = pred_logits.shape[1]  # query 数量（通常是 300）
    
    if total_boxes > 0 and len(cls_losses) > 0:
        num_boxes = max(total_boxes, 1)
        # 🔥 一次性求和，避免逐个累加导致的计算图过长
        total_cls_loss = torch.stack(cls_losses).sum()
        total_bbox_loss = torch.stack(bbox_losses).sum()
        total_giou_loss = torch.stack(giou_losses).sum()
        
        # 🔥 与标准 criterion 保持一致：乘以 Q / num_boxes
        # 标准 criterion 的分类损失公式：loss.mean(1).sum() * Q / num_boxes
        # O2M 和 O2O 都使用相同的缩放方式
        loss_dict['loss_mal'] = total_cls_loss * Q / num_boxes  # 🔥 乘以 Q
        loss_dict['loss_bbox'] = total_bbox_loss / num_boxes
        loss_dict['loss_giou'] = total_giou_loss / num_boxes
    else:
        # 🔥 确保即使没有伪标签，也返回零损失
        # 使用 torch.tensor(0.0) 而不是 pred_logits.sum() * 0.0，避免 NaN 传播
        loss_dict['loss_mal'] = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        loss_dict['loss_bbox'] = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        loss_dict['loss_giou'] = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
    
    # 🔥 添加其他损失键（如果 criterion 需要，设置为 0）
    # 注意：one-to-many 匹配阶段不计算 local loss（loss_fgl, loss_ddf）
    weight_dict = criterion.weight_dict
    for key in weight_dict.keys():
        if key not in loss_dict:
            # 🔥 使用 torch.tensor(0.0) 替代 pred_logits.sum() * 0.0，避免 NaN 传播
            loss_dict[key] = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
    
    # 应用权重
    loss_dict = {k: loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict if k in weight_dict}
    
    return loss_dict


def _compute_loss_with_o2o_matching(
    outputs: Dict,
    targets: List[Dict],
    criterion: nn.Module,
    img_size: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    """
    使用 one-to-one 匈牙利匹配计算损失（warm-up 结束后）
    
    🔥 与 O2M 保持一致的尺度：不乘以 Q，直接除以 num_boxes
    
    Args:
        outputs: 模型输出
        targets: 目标列表（伪标签）
        criterion: 损失计算器（用于获取 matcher 和 num_classes）
        img_size: 图像尺寸 (H, W)
        
    Returns:
        loss_dict: 损失字典
    """
    from ..deim.box_ops import generalized_box_iou, box_iou
    
    pred_logits = outputs['pred_logits']  # [B, Q, C]
    pred_boxes = outputs['pred_boxes']     # [B, Q, 4]
    B, Q, C = pred_logits.shape
    device = pred_logits.device
    
    loss_dict = {}
    
    # 🔥 Step 1: 使用匈牙利匹配获取一对一匹配
    outputs_for_match = {k: v for k, v in outputs.items() if 'aux' not in k and k != 'cross_outputs'}
    hungarian_indices = criterion.matcher(outputs_for_match, targets)['indices']
    
    # 收集损失
    cls_losses = []
    bbox_losses = []
    giou_losses = []
    total_boxes = 0
    
    for b in range(B):
        target = targets[b]
        gt_boxes = target['boxes']  # [N, 4] (cxcywh, normalized)
        gt_labels = target['labels']  # [N]
        num_gts = len(gt_boxes)
        
        if num_gts == 0:
            continue
        
        # 获取匈牙利匹配结果
        src_idx, tgt_idx = hungarian_indices[b]
        
        if len(src_idx) == 0:
            continue
        
        # 确保在正确设备上
        src_idx = src_idx.to(device)
        tgt_idx = tgt_idx.to(device)
        
        # 🔥 分类损失（MAL - Margin-Adaptive Loss）
        target_labels = gt_labels[tgt_idx]  # [M]
        
        # 创建目标类别张量
        target_classes = torch.full((Q,), criterion.num_classes, dtype=torch.long, device=device)
        target_classes[src_idx] = target_labels
        target_onehot = F.one_hot(target_classes, num_classes=criterion.num_classes + 1)[..., :-1]
        
        # 计算 IoU 作为软目标分数
        # 🔥 使用 clone() 创建副本，避免 inplace 操作影响梯度计算
        src_boxes = pred_boxes[b][src_idx].clone()
        tgt_boxes = gt_boxes[tgt_idx].clone()
        
        # 🔥 保护：确保 boxes 有效，避免 NaN（非 inplace 操作）
        src_boxes = torch.nan_to_num(src_boxes, nan=0.0, posinf=1.0, neginf=0.0)
        tgt_boxes = torch.nan_to_num(tgt_boxes, nan=0.0, posinf=1.0, neginf=0.0)
        # 使用 torch.clamp 替代 inplace 赋值
        src_boxes = torch.cat([
            src_boxes[:, :2],  # cx, cy
            src_boxes[:, 2:3].clamp(min=1e-6),  # w
            src_boxes[:, 3:4].clamp(min=1e-6),  # h
        ], dim=-1)
        tgt_boxes = torch.cat([
            tgt_boxes[:, :2],  # cx, cy
            tgt_boxes[:, 2:3].clamp(min=1e-6),  # w
            tgt_boxes[:, 3:4].clamp(min=1e-6),  # h
        ], dim=-1)
        
        H_img, W_img = img_size
        factor = torch.tensor([W_img, H_img, W_img, H_img], device=device, dtype=pred_boxes.dtype).unsqueeze(0)
        src_boxes_abs = box_cxcywh_to_xyxy(src_boxes) * factor
        tgt_boxes_abs = box_cxcywh_to_xyxy(tgt_boxes) * factor
        
        ious, _ = box_iou(src_boxes_abs, tgt_boxes_abs)
        ious = torch.diag(ious).detach().to(dtype=pred_logits.dtype)
        # 🔥 保护：清理 NaN/Inf
        ious = torch.nan_to_num(ious, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 使用 IoU 作为软目标分数
        target_score_o = torch.zeros((Q,), dtype=pred_logits.dtype, device=device)
        target_score_o[src_idx] = ious
        target_score = target_score_o.unsqueeze(-1) * target_onehot
        
        # 计算权重（与标准 MAL 一致）
        pred_score = F.sigmoid(pred_logits[b]).detach()
        if criterion.mal_alpha is not None:
            weight = criterion.mal_alpha * pred_score.pow(criterion.gamma) * (1 - target_onehot) + target_onehot
        else:
            weight = pred_score.pow(criterion.gamma) * (1 - target_onehot) + target_onehot
        
        # BCE 损失
        cls_loss = F.binary_cross_entropy_with_logits(
            pred_logits[b], target_score, weight=weight, reduction='none'
        )
        cls_losses.append(cls_loss.mean(1).sum())
        
        # 🔥 回归损失（L1）
        bbox_loss = F.l1_loss(src_boxes, tgt_boxes, reduction='none').sum(dim=-1)  # [M]
        bbox_losses.append(bbox_loss.sum())
        
        # 🔥 GIoU 损失
        giou = generalized_box_iou(src_boxes_abs, tgt_boxes_abs)
        giou_loss = 1 - torch.diag(giou)
        # 🔥 保护：清理 NaN/Inf
        giou_loss = torch.nan_to_num(giou_loss, nan=0.0, posinf=2.0, neginf=0.0)
        giou_losses.append(giou_loss.sum())
        
        total_boxes += len(src_idx)
    
    # 归一化损失
    Q = pred_logits.shape[1]  # query 数量（通常是 300）
    
    if total_boxes > 0 and len(cls_losses) > 0:
        num_boxes = max(total_boxes, 1)
        # 🔥 一次性求和，避免逐个累加导致的计算图过长
        total_cls_loss = torch.stack(cls_losses).sum()
        total_bbox_loss = torch.stack(bbox_losses).sum()
        total_giou_loss = torch.stack(giou_losses).sum()
        
        # 🔥 与标准 criterion 保持一致：乘以 Q / num_boxes
        # 标准 criterion 的分类损失公式：loss.mean(1).sum() * Q / num_boxes
        # O2M 和 O2O 都使用相同的缩放方式
        loss_dict['loss_mal'] = total_cls_loss * Q / num_boxes  # 🔥 乘以 Q
        loss_dict['loss_bbox'] = total_bbox_loss / num_boxes
        loss_dict['loss_giou'] = total_giou_loss / num_boxes
    else:
        # 🔥 确保即使没有伪标签，也返回零损失
        loss_dict['loss_mal'] = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        loss_dict['loss_bbox'] = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
        loss_dict['loss_giou'] = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
    
    # 🔥 添加其他损失键（如果 criterion 需要，设置为 0）
    # 注意：one-to-one 匹配阶段也不计算 local loss（loss_fgl, loss_ddf）
    weight_dict = criterion.weight_dict
    for key in weight_dict.keys():
        if key not in loss_dict:
            loss_dict[key] = torch.tensor(0.0, device=pred_logits.device, requires_grad=True)
    
    # 应用权重
    loss_dict = {k: loss_dict[k] * weight_dict.get(k, 1.0) for k in loss_dict if k in weight_dict}
    
    return loss_dict


# ============================================================================
# 1. 根据伪标签位置从 Encoder 特征提取 Query 向量
# ============================================================================

class QueryProjector(nn.Module):
    """将 RoI 特征投影到 transformer 查询空间"""
    def __init__(self, in_channels: int = 256, hidden_dim: int = 256, roi_size: int = 7):
        super().__init__()
        # 两层卷积提取空间特征
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 展平后的维度: in_channels * roi_size * roi_size
        flatten_dim = in_channels * roi_size * roi_size
        
        # MLP 投影到查询空间
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_dim, 1024, bias=True)
        self.bn_fc = nn.BatchNorm1d(1024)
        self.relu_fc1 = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(1024, hidden_dim, bias=True)
        self.relu_fc2 = nn.ReLU(inplace=True)
        
    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_feats: [N, C, roi_size, roi_size]
        Returns:
            query: [N, hidden_dim]
        """
        x = self.relu1(self.bn1(self.conv1(roi_feats)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu_fc1(self.bn_fc(self.fc1(x)))
        x = self.relu_fc2(self.fc2(x))
        return x


def extract_queries_from_pseudo_labels(
    encoder_feats: List[torch.Tensor],
    pseudo_targets: List[Dict],
    device: torch.device,
    projector: Optional[nn.Module] = None,
    hidden_dim: int = 256,
    output_size: int = 7,
    img_size: Tuple[int, int] = (800, 800),
    featmap_strides: List[int] = [8, 16, 32],
    dn_number: int = 5,
    return_embeddings: bool = False,  # 🔥 是否返回原始 embeddings（用于原型学习）
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    参考 Semi-DETR 的 prepare_unsup_cdn 函数，根据伪标签位置从 encoder 特征中提取 query 向量
    
    使用 RoI Align 方法：从多尺度特征图中采样
    - 参考 Semi-DETR 原论文的实现
    - 使用 projector 投影到查询空间
    
    Args:
        encoder_feats: 多尺度 encoder 特征 [[B, C, H1, W1], [B, C, H2, W2], [B, C, H3, W3]]
                      例如: [8, 256, 100, 100], [8, 256, 50, 50], [8, 256, 25, 25]
        pseudo_targets: 每张图的伪标签列表
                       [{boxes: [N1, 4], labels: [N1]}, ..., {boxes: [N8, 4], labels: [N8]}]
                       boxes 格式为 cxcywh (归一化坐标，范围 [0, 1])
        device: 设备
        projector: 投影网络 (MLP)，将 RoI 特征投影到查询空间
        hidden_dim: 隐藏维度
        output_size: RoIAlign 输出的空间尺寸
        img_size: 原始图像尺寸 (H, W)，用于坐标转换
        featmap_strides: 各层特征图相对于原图的步长 [8, 16, 32]
        dn_number: 一致性查询的重复组数（默认 5，参考 Semi-DETR）
        return_embeddings: 是否返回原始 embeddings（用于原型学习）
        
    Returns:
        input_query_label: 内容嵌入 [batch_size, pad_size, hidden_dim]
        input_query_bbox: 位置嵌入 [batch_size, pad_size, 4]（inverse_sigmoid 空间）
        attn_mask: 注意力掩码 [pad_size, pad_size]（用于组间屏蔽）
        dn_meta: 元信息字典，包含：
                - pad_size: 一致性段长度
                - single_pad: 每组长度
                - num_dn_group: 一致性组数
                - known_bid: batch 索引
                - map_known_indice: 索引映射
                - loss_weights: 损失权重
                - box_counts: 每张图的框数量
    """
    from torchvision.ops import roi_align
    
    if encoder_feats is None or len(encoder_feats) == 0:
        return None, None, None, None
    
    B = encoder_feats[0].shape[0]
    C = encoder_feats[0].shape[1]
    num_levels = len(encoder_feats)
    H_img, W_img = img_size
    
    # ========== Step 1: 收集所有图片的伪标签框（参考 Semi-DETR）==========
    all_boxes_abs = []      # 绝对坐标 xyxy（用于 RoIAlign）
    all_boxes_norm = []     # 归一化坐标 cxcywh（用于位置嵌入）
    all_batch_ids = []      # batch 索引
    box_counts = []         # 每张图的框数量
    loss_weights_list = []  # 每个框的损失权重（空伪标签使用兜底框时权重为 0）
    
    for b in range(B):
        boxes = pseudo_targets[b].get('boxes', None)
        
        if boxes is None or len(boxes) == 0:
            fallback_box_norm = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device, dtype=torch.float32)
            fallback_box_abs = torch.tensor([[W_img / 4, H_img / 4, 3 * W_img / 4, 3 * H_img / 4]], 
                                            device=device, dtype=torch.float32)
            all_boxes_norm.append(fallback_box_norm)
            all_boxes_abs.append(fallback_box_abs)
            all_batch_ids.append(torch.full((1,), b, dtype=torch.long, device=device))
            box_counts.append(1)
            loss_weights_list.append(torch.zeros(1, device=device))
        else:
            num_boxes = boxes.shape[0]
            box_counts.append(num_boxes)
            
            boxes_cxcywh_norm = boxes.clone().clamp(min=0.0, max=1.0)
            all_boxes_norm.append(boxes_cxcywh_norm)
            
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = (cx - w / 2) * W_img
            y1 = (cy - h / 2) * H_img
            x2 = (cx + w / 2) * W_img
            y2 = (cy + h / 2) * H_img
            boxes_xyxy_abs = torch.stack([x1, y1, x2, y2], dim=-1)
            
            all_boxes_abs.append(boxes_xyxy_abs)
            all_batch_ids.append(torch.full((num_boxes,), b, dtype=torch.long, device=device))
            loss_weights_list.append(torch.ones(num_boxes, device=device))
    
    batched_boxes_abs = torch.cat(all_boxes_abs, dim=0)
    batched_boxes_norm = torch.cat(all_boxes_norm, dim=0)
    batched_batch_ids = torch.cat(all_batch_ids, dim=0)
    batched_loss_weights = torch.cat(loss_weights_list, dim=0)
    total_boxes = batched_boxes_abs.shape[0]
    
    # ========== Step 2: 使用 RoI Align 提取特征（优化版：单尺度 + 清理无效值）==========
    feat_dtype = encoder_feats[0].dtype
    
    # 清理无效框（NaN, Inf, 零尺寸）
    batched_boxes_abs = torch.nan_to_num(batched_boxes_abs, nan=0.0, posinf=W_img, neginf=0.0)
    batched_boxes_abs = batched_boxes_abs.clamp(min=0.0)
    batched_boxes_abs[:, 2] = batched_boxes_abs[:, 2].clamp(max=W_img)
    batched_boxes_abs[:, 3] = batched_boxes_abs[:, 3].clamp(max=H_img)
    # 确保 x2 > x1, y2 > y1（最小 1 像素）
    batched_boxes_abs[:, 2] = torch.max(batched_boxes_abs[:, 2], batched_boxes_abs[:, 0] + 1)
    batched_boxes_abs[:, 3] = torch.max(batched_boxes_abs[:, 3], batched_boxes_abs[:, 1] + 1)
    
    # 使用中间尺度特征（level=1）作为默认，更稳定
    default_level = min(1, num_levels - 1)
    stride = featmap_strides[default_level]
    spatial_scale = 1.0 / stride
        
    # 构建 ROIs [N, 5] = [batch_id, x1, y1, x2, y2]
    rois = torch.cat([batched_batch_ids.unsqueeze(-1).float(), batched_boxes_abs], dim=-1)
    rois = rois.to(dtype=feat_dtype)
        
    # 单次 roi_align 调用（避免循环内同步）
    roi_feats_all = roi_align(
        encoder_feats[default_level],
            rois,
            output_size=(output_size, output_size),
            spatial_scale=spatial_scale,
            sampling_ratio=0,
            aligned=True
        )
    
    if projector is not None:
        roi_feats_fp32 = roi_feats_all.float()
        query_feats = projector(roi_feats_fp32)
    else:
        query_feats = roi_feats_all.float().mean(dim=[2, 3])
    
    single_pad = int(max(box_counts)) if box_counts else 1
    pad_size = single_pad * dn_number
    
    known_bid = batched_batch_ids.repeat(dn_number, 1).view(-1)
    known_boxes_norm = batched_boxes_norm.repeat(dn_number, 1)
    known_query_feats = query_feats.repeat(dn_number, 1)
    
    consistency_bbox_embed = inverse_sigmoid(known_boxes_norm)
    
    input_query_label = torch.zeros(B, pad_size, hidden_dim, device=device)
    input_query_bbox = torch.zeros(B, pad_size, 4, device=device)
    
    map_known_indice = torch.tensor([], device=device, dtype=torch.long)
    if box_counts:
        map_known_indice = torch.cat([torch.arange(num, device=device) for num in box_counts])
        map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(dn_number)]).long()
    
    if len(known_bid) > 0 and len(map_known_indice) > 0:
        # 🔥 确保类型匹配（混合精度训练时 known_query_feats 可能是 float16）
        input_query_label[(known_bid.long(), map_known_indice)] = known_query_feats.to(input_query_label.dtype)
        input_query_bbox[(known_bid.long(), map_known_indice)] = consistency_bbox_embed.to(input_query_bbox.dtype)
    
    loss_weights = batched_loss_weights.unsqueeze(-1).repeat(dn_number, 1)
    
    # 🔥 构造 attn_mask：组间屏蔽 + padding 位置屏蔽（向量化优化版）
    attn_mask = torch.zeros(pad_size, pad_size, device=device).bool()
    
    # 1. 组间屏蔽：每组只能看到自己组内的成员（向量化）
    # 创建组索引
    group_ids = torch.arange(pad_size, device=device) // single_pad  # [pad_size]
    # 组间屏蔽：不同组之间不可见
    attn_mask = group_ids.unsqueeze(0) != group_ids.unsqueeze(1)  # [pad_size, pad_size]
    
    # 🔥 2. Padding 位置屏蔽（向量化优化，避免 GPU/CPU 同步）
    # 🔥 直接在 CPU 上计算有效位置（box_counts 已经是 Python list）
    # 这样完全避免了 .any() 和布尔索引导致的 GPU/CPU 同步
    
    # 🔥 在 CPU 上计算有效位置的索引（纯 Python，无同步）
    valid_positions_cpu = []
    for group_idx in range(dn_number):
        group_offset = group_idx * single_pad
        for b, count in enumerate(box_counts):
            for pos in range(count):
                valid_positions_cpu.append(group_offset + pos)
    
    # 🔥 使用掩码张量替代布尔索引（避免同步）
    # 方法：创建全 True 的 invalid_mask，然后在有效位置设为 False
    invalid_mask_1d = torch.ones(pad_size, device=device, dtype=torch.bool)
    if len(valid_positions_cpu) > 0:
        valid_indices = torch.tensor(valid_positions_cpu, device=device, dtype=torch.long)
        invalid_mask_1d[valid_indices] = False
    
    # 🔥 使用 torch.where 替代布尔索引赋值（全 GPU 操作，无同步）
    # attn_mask[:, invalid] = True 等价于：
    # attn_mask = torch.where(invalid_mask_1d.unsqueeze(0), True_tensor, attn_mask)
    invalid_col_mask = invalid_mask_1d.unsqueeze(0).expand(pad_size, -1)  # [pad_size, pad_size]
    invalid_row_mask = invalid_mask_1d.unsqueeze(1).expand(-1, pad_size)  # [pad_size, pad_size]
    
    # 合并屏蔽：无效列 OR 无效行
    padding_mask = invalid_col_mask | invalid_row_mask
    attn_mask = attn_mask | padding_mask
    
    # 🔥 对角线允许自己看自己（避免全 -inf）
    # 使用 scatter_ 替代布尔索引
    diag_indices = torch.arange(pad_size, device=device)
    # 创建对角线掩码：只在 invalid 位置的对角线上设为 False
    diag_false_mask = torch.zeros(pad_size, pad_size, device=device, dtype=torch.bool)
    diag_false_mask[diag_indices, diag_indices] = True
    # 只在 invalid 对角线位置取消屏蔽
    invalid_diag_mask = diag_false_mask & invalid_row_mask
    attn_mask = attn_mask & ~invalid_diag_mask
    
    dn_meta = {
        'pad_size': pad_size,
        'single_pad': single_pad,
        'num_dn_group': dn_number,
        'known_bid': known_bid,
        'map_known_indice': map_known_indice,
        'loss_weights': loss_weights,
        'box_counts': box_counts,
    }
    
    # 🔥 如果需要返回 embeddings（用于语义原型学习）
    if return_embeddings:
        # 收集每个框的 labels
        all_labels = []
        for b in range(B):
            labels = pseudo_targets[b].get('labels', None)
            if labels is None or len(labels) == 0:
                # 空伪标签使用的兜底框，label 设为 -1（无效）
                all_labels.append(torch.tensor([-1], device=device, dtype=torch.long))
            else:
                all_labels.append(labels)
        batched_labels = torch.cat(all_labels, dim=0)  # [total_boxes]
        
        dn_meta['query_embeddings'] = query_feats  # [total_boxes, hidden_dim]（未重复）
        dn_meta['embedding_labels'] = batched_labels  # [total_boxes]
        dn_meta['embedding_batch_ids'] = batched_batch_ids  # [total_boxes]
        dn_meta['embedding_loss_weights'] = batched_loss_weights  # [total_boxes]（兜底框权重为0）
    
    return input_query_label, input_query_bbox, attn_mask, dn_meta



def compute_semi_detr_unsup_loss(
    student_model: nn.Module,
    teacher_model: nn.Module,
    criterion: nn.Module,
    images_weak: torch.Tensor,
    images_strong: torch.Tensor,
    metas: Dict,
    args=None,
    teacher_outputs: Optional[Dict] = None,
    unsup_weight: float = 4.0,
    curr_epoch: int = 0,
    scaler=None,
    device: torch.device = None,
    transform_weak: Optional[Dict] = None,
    transform_strong: Optional[Dict] = None,
    img_size_weak: Optional[Tuple[int, int]] = None,
    img_size_strong: Optional[Tuple[int, int]] = None,
    vdvi_weak: Optional[torch.Tensor] = None,
    query_projector: Optional[nn.Module] = None,
    visualize_pseudo_labels: bool = False,
    visualize_save_dir: Optional[str] = None,
    visualize_class_names: Optional[List[str]] = None,
    adaptive_threshold_manager: Optional['AdaptiveThresholdManager'] = None,
    # 🔥 以下参数保留用于向后兼容，但优先从 cfg 读取
    conf_thresh: Optional[float] = None,
    consistency_weight: Optional[float] = None,
    warm_up_epochs: Optional[int] = None,
    use_hybrid_matching: Optional[bool] = None,
    hybrid_topk: Optional[int] = None,
    vdvi_thresh: Optional[float] = None,
    use_vdvi_filter: Optional[bool] = None,
    pseudo_nms_iou: Optional[float] = None,
    pseudo_nms_containment: Optional[float] = None,
    use_adaptive_threshold: Optional[bool] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    """
    计算半监督无监督损失（简化版，适用于单卡训练）
    """
    images_weak_rgb = images_weak['rgb']
    images_weak_hsv = images_weak.get('hsv', None)
    images_strong_rgb = images_strong['rgb']
    images_strong_hsv = images_strong.get('hsv', None)
    
    if device is None:
        device = images_weak_rgb.device
    
    # 🔥 从配置文件读取超参数（如果参数为 None，则从 cfg 读取）
    if args is not None and hasattr(args, 'yaml_cfg'):
        cfg = args.yaml_cfg
        conf_thresh = conf_thresh if conf_thresh is not None else cfg.get('conf_thresh', 0.4)
        consistency_weight = consistency_weight if consistency_weight is not None else cfg.get('ssl_consistency_weight', 10.0)
        warm_up_epochs = warm_up_epochs if warm_up_epochs is not None else cfg.get('ssl_warm_up_epochs', 100)
        use_hybrid_matching = use_hybrid_matching if use_hybrid_matching is not None else cfg.get('ssl_use_hybrid_matching', True)
        hybrid_topk = hybrid_topk if hybrid_topk is not None else cfg.get('ssl_hybrid_topk', 13)
        vdvi_thresh = vdvi_thresh if vdvi_thresh is not None else cfg.get('ssl_vdvi_thresh', 0.030)
        use_vdvi_filter = use_vdvi_filter if use_vdvi_filter is not None else cfg.get('ssl_use_vdvi_filter', True)
        pseudo_nms_iou = pseudo_nms_iou if pseudo_nms_iou is not None else cfg.get('ssl_pseudo_nms_iou', 0.5)
        pseudo_nms_containment = pseudo_nms_containment if pseudo_nms_containment is not None else cfg.get('ssl_pseudo_nms_containment', 0.8)
        use_adaptive_threshold = use_adaptive_threshold if use_adaptive_threshold is not None else cfg.get('ssl_use_adaptive_threshold', False)
        # 🔥 语义原型筛选参数
        use_proto_filter = cfg.get('ssl_use_proto_filter', False)
        proto_filter_threshold = cfg.get('ssl_proto_filter_threshold', 0.5)
        proto_filter_start_epoch = cfg.get('ssl_proto_filter_start_epoch', 5)
    else:
        conf_thresh = conf_thresh if conf_thresh is not None else 0.4
        consistency_weight = consistency_weight if consistency_weight is not None else 10.0
        warm_up_epochs = warm_up_epochs if warm_up_epochs is not None else 100
        use_hybrid_matching = use_hybrid_matching if use_hybrid_matching is not None else True
        hybrid_topk = hybrid_topk if hybrid_topk is not None else 13
        vdvi_thresh = vdvi_thresh if vdvi_thresh is not None else 0.030
        use_vdvi_filter = use_vdvi_filter if use_vdvi_filter is not None else True
        pseudo_nms_iou = pseudo_nms_iou if pseudo_nms_iou is not None else 0.5
        pseudo_nms_containment = pseudo_nms_containment if pseudo_nms_containment is not None else 0.8
        use_adaptive_threshold = use_adaptive_threshold if use_adaptive_threshold is not None else False
        # 🔥 语义原型筛选参数（默认值）
        use_proto_filter = False
        proto_filter_threshold = 0.5
        proto_filter_start_epoch = 5
    
    B = images_weak_rgb.shape[0]
    
    # 🔥 单卡训练：空 batch 直接返回零损失
    if B == 0:
        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return dummy_loss, {}, {'num_pseudo_boxes': 0, 'consistency_loss': 0.0}
    
    H, W = images_weak_rgb.shape[2:]
    
    stats = {'num_pseudo_boxes': 0}
    
    teacher_logits = teacher_outputs['pred_logits']
    teacher_boxes_weak = teacher_outputs['pred_boxes']
    teacher_encoder_feats = teacher_outputs.get('encoder_feate', None)
    
    adaptive_thresh_list = []
    
    # NaN/Inf 会在后续通过 nan_to_num 处理，这里不进行同步检查
    
    pseudo_targets = []
    teacher_pseudo_targets = []
    total_pseudo_boxes = 0
    adaptive_thresh_list = []
    
    all_probs = teacher_logits.sigmoid()
    all_scores, all_labels = all_probs.max(dim=-1)
    
    empty_boxes = torch.zeros((0, 4), device=device)
    empty_labels = torch.zeros((0,), dtype=torch.long, device=device)
    empty_scores = torch.zeros((0,), device=device)
    
    for b in range(B):
        scores = all_scores[b]  
        labels = all_labels[b]  
        
        if use_adaptive_threshold and adaptive_threshold_manager is not None and scores.numel() >= 2:
            mu, sigma = adaptive_threshold_manager._fit_lognormal_mle(scores)
            adaptive_thresh = adaptive_threshold_manager._compute_threshold_from_lognormal(scores, mu, sigma)
            keep = scores >= adaptive_thresh
            adaptive_thresh_list.append(adaptive_thresh.detach())
        else:
            keep = scores >= conf_thresh
        
        # 🔥 优化：直接使用布尔索引获取数据
        kept_boxes_weak = teacher_boxes_weak[b][keep]
        kept_labels = labels[keep]
        kept_scores = scores[keep]
            
        num_kept = kept_boxes_weak.shape[0]
            
        if num_kept > 0:
            # 🌿 Step 2: VDVI 植被指数筛选（固定阈值）
            if use_vdvi_filter and vdvi_weak is not None and kept_boxes_weak.shape[0] > 0:
                vdvi_map = vdvi_weak[b, 0]
                vdvi_keep, vdvi_scores_batch = filter_pseudo_labels_by_vdvi(
                    boxes=kept_boxes_weak,
                    vdvi_map=vdvi_map,
                    vdvi_thresh=vdvi_thresh,
                    box_format='cxcywh',
                )
                kept_boxes_weak = kept_boxes_weak[vdvi_keep]
                kept_labels = kept_labels[vdvi_keep]
                kept_scores = kept_scores[vdvi_keep]
            
            # 🔥 Step 3: 伪标签 NMS（跨类别，消除 small/big 嵌套与重复框）
            # 使用 IoU + 包含度检测，解决"大框套小框"问题
            if pseudo_nms_iou is not None and pseudo_nms_iou > 0 and kept_boxes_weak.shape[0] > 1:
                nms_keep = cross_class_nms(
                    boxes=kept_boxes_weak,
                    labels=kept_labels,
                    scores=kept_scores,
                    iou_threshold=float(pseudo_nms_iou),
                    containment_threshold=float(pseudo_nms_containment),
                )
                kept_boxes_weak = kept_boxes_weak[nms_keep]
                kept_labels = kept_labels[nms_keep]
                kept_scores = kept_scores[nms_keep]
            
            # 如果筛选后没有剩余伪标签
            num_kept = kept_boxes_weak.shape[0]
            if num_kept == 0:
                pseudo_targets.append({'boxes': empty_boxes, 'labels': empty_labels, 'weights': torch.zeros(0, device=device)})
                teacher_pseudo_targets.append({'boxes': empty_boxes, 'labels': empty_labels, 'scores': empty_scores})
                continue
            
            # 🔥 Step 4: 语义原型筛选（与类别原型比较，过滤低相似度伪标签）
            if use_proto_filter and curr_epoch >= proto_filter_start_epoch and teacher_encoder_feats is not None:
                # 从 teacher encoder 特征中提取伪标签区域的特征，与类别原型比较
                proto_keep = _filter_by_prototype_similarity(
                    encoder_feats=teacher_encoder_feats,
                    boxes=kept_boxes_weak,
                    labels=kept_labels,
                    batch_idx=b,
                    threshold=proto_filter_threshold,
                    img_size=(H, W),
                )
                kept_indices = torch.where(proto_keep)[0]
                if kept_indices.numel() > 0:
                    kept_boxes_weak = kept_boxes_weak[kept_indices]
                    kept_labels = kept_labels[kept_indices]
                    kept_scores = kept_scores[kept_indices]
                    num_kept = kept_boxes_weak.shape[0]
                    stats['train_proto_filtered'] = stats.get('train_proto_filtered', 0) + (len(proto_keep) - kept_indices.numel())
                else:
                    pseudo_targets.append({'boxes': empty_boxes, 'labels': empty_labels, 'weights': torch.zeros(0, device=device)})
                    teacher_pseudo_targets.append({'boxes': empty_boxes, 'labels': empty_labels, 'scores': empty_scores})
                    stats['train_proto_filtered'] = stats.get('train_proto_filtered', 0) + len(proto_keep)
                    continue
            
            # 设置权重（原型筛选后的伪标签权重为1）
            u_weights = torch.ones(num_kept, device=device)
            
            teacher_pseudo_targets.append({
                'boxes': kept_boxes_weak.detach(),
                'labels': kept_labels.detach(),
                'scores': kept_scores.detach(),
            })
            
            if transform_weak is not None and transform_strong is not None and \
               img_size_weak is not None and img_size_strong is not None:
                kept_boxes = transform_boxes_cxcywh_weak_to_strong(
                    kept_boxes_weak,
                    transform_weak,
                    transform_strong,
                    img_size_weak=img_size_weak,
                    img_size_strong=img_size_strong
                )
            else:
                kept_boxes = kept_boxes_weak
            
            total_pseudo_boxes += num_kept
            
            pseudo_targets.append({
                'boxes': kept_boxes.detach(),
                'labels': kept_labels.detach(),
                'weights': u_weights.detach(), 
            })
        else:
            pseudo_targets.append({'boxes': empty_boxes, 'labels': empty_labels, 'weights': torch.zeros(0, device=device)})
            teacher_pseudo_targets.append({'boxes': empty_boxes, 'labels': empty_labels, 'scores': empty_scores})
    
    stats['num_pseudo_boxes'] = total_pseudo_boxes
    
    if use_adaptive_threshold and len(adaptive_thresh_list) > 0:
        thresh_tensor = torch.stack(adaptive_thresh_list)
        stats['adaptive_thresh'] = float(thresh_tensor.mean().cpu())
        stats['adaptive_samples'] = len(adaptive_thresh_list)
    
    if visualize_pseudo_labels and visualize_save_dir is not None and total_pseudo_boxes > 0:
        try:
            for b_idx, target in enumerate(teacher_pseudo_targets):
                if target is not None and target.get('boxes') is not None and target['boxes'].shape[0] > 0:
                    if isinstance(images_weak, dict):
                        if 'rgb' in images_weak:
                            img_b = {'rgb': images_weak['rgb'][b_idx:b_idx+1]}
                            if 'hsv' in images_weak and images_weak['hsv'] is not None:
                                img_b['hsv'] = images_weak['hsv'][b_idx:b_idx+1]
                        else:
                            img_b = images_weak
                    else:
                        img_b = images_weak[b_idx:b_idx+1] if images_weak.dim() == 4 else images_weak
                    
                    boxes = target['boxes']
                    labels = target['labels']
                    scores = target.get('scores', torch.ones_like(labels, dtype=torch.float32))
                    
                    save_path = f"{visualize_save_dir}/pseudo_labels_epoch_{curr_epoch:03d}_b{b_idx:02d}.jpg"
                    visualize_pseudo_labels_during_training(
                        images_weak=img_b,
                        pseudo_boxes=boxes,
                        pseudo_labels=labels,
                        pseudo_scores=scores,
                        save_path=save_path,
                        class_names=visualize_class_names,
                        title=f"Step {curr_epoch} B{b_idx} | conf>={conf_thresh:.2f} | VDVI={use_vdvi_filter} | NMS={pseudo_nms_iou:.2f}",
                    )
        except Exception as e:
            print(f"⚠️ 可视化伪标签时出错（不影响训练）: {e}")
    
    student_cross_query_label = None
    student_cross_query_bbox = None
    student_cross_attn_mask = None
    student_cross_meta = None
    
    if teacher_encoder_feats is not None and total_pseudo_boxes > 0:
        student_cross_query_label, student_cross_query_bbox, student_cross_attn_mask, student_cross_meta = extract_queries_from_pseudo_labels(
            encoder_feats=teacher_encoder_feats,
            pseudo_targets=teacher_pseudo_targets,
            device=device,
            projector=query_projector,
            img_size=img_size_weak,
            dn_number=5,
        )
    
    student_model.train()
    teacher_model.eval()
    
    if total_pseudo_boxes == 0:
        
        zero_loss = torch.tensor(0.0, device=device)
        is_ddp = isinstance(student_model, torch.nn.parallel.DistributedDataParallel)
        
        if is_ddp:
            for p in student_model.parameters():
                if p.requires_grad:
                    zero_loss = zero_loss + p.sum()
            zero_loss = zero_loss * 0.0
        else:
            p = next(student_model.parameters())
            zero_loss = (p.sum() * 0.0).to(device)
            if not zero_loss.requires_grad and p.requires_grad:
                 zero_loss.requires_grad_(True)
        
        loss_dict_unsup = {
            'unsup_loss_mal': zero_loss,
            'unsup_loss_fgl': zero_loss * 0,
            'unsup_loss_ddf': zero_loss * 0,
            'unsup_consistency': zero_loss * 0,
        }
        stats['skip_reason'] = 'no_pseudo_boxes'
        return zero_loss, loss_dict_unsup, stats
    use_o2m = use_hybrid_matching and curr_epoch < warm_up_epochs  
    
    if scaler is not None:
        with torch.amp.autocast(device_type='cuda', enabled=True):
            student_outputs = student_model(
                images_strong_rgb,
                targets=pseudo_targets,
                hsv=images_strong_hsv,
                cross_query_label=student_cross_query_label,
                cross_query_bbox=student_cross_query_bbox,
                cross_attn_mask=student_cross_attn_mask,
                cross_meta=student_cross_meta,
            )
        with torch.amp.autocast(device_type='cuda', enabled=False):
            if use_o2m:
                loss_dict_raw = _compute_loss_with_o2m_matching(
                    student_outputs, pseudo_targets, criterion, img_size_strong,
                    hybrid_topk=hybrid_topk,
                    beta=2.0, 
                )
            else:
                loss_dict_raw = _compute_loss_with_o2o_matching(
                    student_outputs, pseudo_targets, criterion, img_size_strong,
                )
    else:
        student_outputs = student_model(
            images_strong_rgb,
            targets=pseudo_targets,
            hsv=images_strong_hsv,
            cross_query_label=student_cross_query_label,
            cross_query_bbox=student_cross_query_bbox,
            cross_attn_mask=student_cross_attn_mask,
            cross_meta=student_cross_meta,
        )
        if use_o2m:
            loss_dict_raw = _compute_loss_with_o2m_matching(
                student_outputs, pseudo_targets, criterion, img_size_strong,
                hybrid_topk=hybrid_topk,
                beta=2.0,  
            )
        else:
            loss_dict_raw = _compute_loss_with_o2o_matching(
                student_outputs, pseudo_targets, criterion, img_size_strong,
            )
    
    loss_pseudo = sum(loss_dict_raw.values())
    loss_pseudo_scaled = loss_pseudo
    loss_consistency = loss_pseudo * 0.0
    
    
    should_compute_consistency = (consistency_weight > 0) and (curr_epoch < warm_up_epochs)  # 🔥 改为基于 epoch
    
    student_cross_outputs = student_outputs.get('cross_outputs', None) if should_compute_consistency else None
    student_encoder_feats = student_outputs.get('encoder_feate', None) if should_compute_consistency else None
    
    if should_compute_consistency and student_cross_outputs is not None and student_cross_meta is not None and student_encoder_feats is not None:
        student_cross_dec_hs = student_cross_outputs.get('dec_hs', None)
        
        if student_cross_dec_hs is not None:
            teacher_cross_query_label, teacher_cross_query_bbox, teacher_cross_attn_mask, teacher_cross_meta = extract_queries_from_pseudo_labels(
                encoder_feats=student_encoder_feats,
                pseudo_targets=pseudo_targets,
                device=device,
                projector=query_projector,
                img_size=img_size_strong,
                dn_number=5,
            )
            
            if teacher_cross_query_label is not None:
                teacher_cross_query_label = teacher_cross_query_label.float()
            if teacher_cross_query_bbox is not None:
                teacher_cross_query_bbox = teacher_cross_query_bbox.float()
            
            with torch.no_grad():
                # 🔥 使用与学生模型相同的 AMP 设置
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        teacher_outputs_with_cross = teacher_model(
                            images_weak_rgb,
                            targets=teacher_pseudo_targets,
                            is_teacher=True,
                            hsv=images_weak_hsv,
                            cross_query_label=teacher_cross_query_label,
                            cross_query_bbox=teacher_cross_query_bbox,
                            cross_attn_mask=teacher_cross_attn_mask,
                            cross_meta=teacher_cross_meta,
                        )
                else:
                    teacher_outputs_with_cross = teacher_model(
                        images_weak_rgb,
                        targets=teacher_pseudo_targets,
                        is_teacher=True,
                        hsv=images_weak_hsv,
                        cross_query_label=teacher_cross_query_label,
                        cross_query_bbox=teacher_cross_query_bbox,
                        cross_attn_mask=teacher_cross_attn_mask,
                        cross_meta=teacher_cross_meta,
                    )
            
            teacher_cross_outputs = teacher_outputs_with_cross.get('cross_outputs', None)
            
            if teacher_cross_outputs is not None:
                teacher_cross_dec_hs = teacher_cross_outputs.get('dec_hs', None)
                
                if teacher_cross_dec_hs is not None:
                    consistency_losses = []
                    
                    known_bid = student_cross_meta.get('known_bid', None)
                    map_known_indice = student_cross_meta.get('map_known_indice', None)
                    loss_weights = student_cross_meta.get('loss_weights', None)
                    
                    if loss_weights is not None and curr_epoch >= warm_up_epochs:  # 🔥 改为基于 epoch
                        loss_weights = torch.zeros_like(loss_weights)
                    
                    def safe_normalize_and_loss_with_index(s_hs, t_hs, known_bid, map_known_indice, loss_weights):
                        if known_bid is None or map_known_indice is None or known_bid.numel() == 0:
                            return None
                        
                        h1 = s_hs[known_bid.long(), map_known_indice]
                        h2 = t_hs[known_bid.long(), map_known_indice]
                        
                        if h1.shape[0] == 0:
                            return None
                        
                        # L2 归一化（参考 Semi-DETR 第 550 行）
                        h1_norm = F.normalize(h1, p=2, dim=-1, eps=1e-6)
                        h2_norm = F.normalize(h2, p=2, dim=-1, eps=1e-6)
                        
                        # 使用 nan_to_num 避免 GPU/CPU 同步
                        h1_norm = torch.nan_to_num(h1_norm, nan=0.0, posinf=0.0, neginf=0.0)
                        h2_norm = torch.nan_to_num(h2_norm, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        if loss_weights is not None and loss_weights.numel() > 0:
                            # loss_weights: [total_valid, 1]
                            mse_per_query = F.mse_loss(h1_norm, h2_norm.detach(), reduction='none')  # [total_valid, C]
                            weighted_mse = mse_per_query * loss_weights  # [total_valid, C]
                            loss = weighted_mse.mean()
                        else:
                            loss = F.mse_loss(h1_norm, h2_norm.detach(), reduction='mean')
                        
                        return loss
                    
                    if isinstance(student_cross_dec_hs, dict) and isinstance(teacher_cross_dec_hs, dict):
                        layer_indices = set(student_cross_dec_hs.keys()) & set(teacher_cross_dec_hs.keys())
                        for layer_idx in sorted(layer_indices):
                            loss = safe_normalize_and_loss_with_index(
                                student_cross_dec_hs[layer_idx],
                                teacher_cross_dec_hs[layer_idx],
                                known_bid,
                                map_known_indice,
                                loss_weights,
                            )
                            if loss is not None:
                                consistency_losses.append(loss)
                    
                    elif isinstance(student_cross_dec_hs, (list, tuple)) and isinstance(teacher_cross_dec_hs, (list, tuple)):
                        num_layers = min(len(student_cross_dec_hs), len(teacher_cross_dec_hs))
                        for layer_idx in range(num_layers):
                            loss = safe_normalize_and_loss_with_index(
                                student_cross_dec_hs[layer_idx],
                                teacher_cross_dec_hs[layer_idx],
                                known_bid,
                                map_known_indice,
                                loss_weights,
                            )
                            if loss is not None:
                                consistency_losses.append(loss)
                    
                    if len(consistency_losses) > 0:
                        loss_consistency = torch.stack(consistency_losses).mean()
    
                    del consistency_losses
                
    if curr_epoch >= warm_up_epochs:  
        effective_consistency_weight = 0.0
    else:
        effective_consistency_weight = consistency_weight
    
    total_loss = unsup_weight * loss_pseudo_scaled + effective_consistency_weight * loss_consistency
    loss_dict = {f"unsup_{k}": v * unsup_weight for k, v in loss_dict_raw.items()}
    loss_dict['unsup_consistency'] = loss_consistency * effective_consistency_weight
    
    return total_loss, loss_dict, stats

