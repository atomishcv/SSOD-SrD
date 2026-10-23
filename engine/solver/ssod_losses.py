"""
Semi-DETR 风格的半监督损失计算模块

将监督损失和无监督损失的计算逻辑分离到独立的函数中，
便于维护和调试。

包含：
1. compute_sup_loss: 监督损失
2. compute_unsup_loss: 无监督损失（伪标签）
3. compute_unsup_loss_with_consistency: 无监督损失 + 跨视角一致性损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ..misc import dist_utils


def compute_sup_loss(
    student_model: nn.Module,
    criterion: nn.Module,
    images: torch.Tensor,
    targets: List[Dict],
    metas: Dict,
    scaler=None,
    # 🔥 从 cfg 读取参数
    args=None,
    current_epoch: int = 0,
    # 🔥 以下参数保留用于向后兼容，但优先从 cfg 读取
    warm_up_epochs: Optional[int] = None,
    use_hybrid_matching: Optional[bool] = None,
    hybrid_topk: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    计算监督损失（Semi-DETR 风格）
    
    🔥 关键修复：监督分支始终使用 one-to-one 匹配（标准 Hungarian 匹配）
    只有无监督分支才在 warm-up 阶段使用 one-to-many 匹配。
    
    Semi-DETR 原始设计：
    - 监督分支：始终 one-to-one（GT 是准确的，不需要 one-to-many）
    """
    # 🔥 从配置文件读取超参数（如果参数为 None，则从 cfg 读取）
    if args is not None and hasattr(args, 'yaml_cfg'):
        cfg = args.yaml_cfg
        warm_up_epochs = warm_up_epochs if warm_up_epochs is not None else cfg.get('ssl_warm_up_epochs', 100)
        use_hybrid_matching = use_hybrid_matching if use_hybrid_matching is not None else cfg.get('ssl_use_hybrid_matching', False)
        hybrid_topk = hybrid_topk if hybrid_topk is not None else cfg.get('ssl_hybrid_topk', 13)
    else:
        warm_up_epochs = warm_up_epochs if warm_up_epochs is not None else 100
        use_hybrid_matching = use_hybrid_matching if use_hybrid_matching is not None else False
        hybrid_topk = hybrid_topk if hybrid_topk is not None else 13
    
    """
    - 无监督分支：warm-up 阶段 one-to-many，之后 one-to-one
    
    Args:
        student_model: 学生模型
        criterion: 损失函数
        images: 有标注图像 [B, C, H, W] 或字典 {'rgb': tensor, 'hsv': tensor}
        targets: 标注目标列表
        metas: 元信息（epoch, step 等）
        scaler: AMP scaler（可选）
        current_epoch: 当前 epoch 数（监督分支不使用）
        warm_up_epochs: warm-up epoch 数（监督分支不使用）
        use_hybrid_matching: 是否启用混合匹配（监督分支忽略此参数，始终 one-to-one）
        hybrid_topk: one-to-many 匹配时每个 GT 分配的 query 数量（监督分支不使用）
    
    Returns:
        loss_sum: 监督损失总和
        loss_dict: 各项损失字典（带 'sup_' 前缀）
    """
    student_model.train()
    
    # 🔥 处理字典格式的图像（HSV分支）或普通tensor
    if isinstance(images, dict):
        # 字典格式：{'rgb': tensor, 'hsv': tensor}
        images_rgb = images['rgb']
        images_hsv = images.get('hsv', None)  # 可选
    else:
        # 普通tensor格式
        images_rgb = images
        images_hsv = None
    
    # 输出提示（仅首次）
    if not hasattr(compute_sup_loss, '_logged'):
        if dist_utils.is_main_process():
            print(f"✅ 监督训练始终使用 One-to-One 匹配（Semi-DETR 标准设计）")
        compute_sup_loss._logged = True
    
    # 获取图像尺寸（用于 one-to-many 匹配）
    # 注意：orig_size 格式是 [width, height]，需要转换为 (H, W)
    if len(targets) > 0 and 'orig_size' in targets[0]:
        # 从 target 中获取原始图像尺寸 [W, H] -> (H, W)
        orig_size = targets[0]['orig_size'].tolist()
        img_size = (orig_size[1], orig_size[0])  # (H, W)
    elif len(targets) > 0 and 'size' in targets[0]:
        # size 格式可能是 [W, H] 或 [H, W]，需要检查
        size = targets[0]['size'].tolist()
        if len(size) == 2:
            # 假设是 [W, H]，转换为 (H, W)
            img_size = (size[1], size[0])
        else:
            img_size = tuple(size)
    else:
        # 从图像 tensor 推断尺寸
        img_size = (images_rgb.shape[2], images_rgb.shape[3])  # (H, W)
    
    if scaler is not None:
        with torch.amp.autocast(device_type='cuda', enabled=True):
            outputs = student_model(images_rgb, targets=targets, hsv=images_hsv)
        with torch.amp.autocast(device_type='cuda', enabled=False):
                # Warm-up 之后：使用标准 one-to-one 匹配
            loss_dict_raw = criterion(outputs, targets, **metas)
    else:
        outputs = student_model(images_rgb, targets=targets, hsv=images_hsv)
        # Warm-up 之后：使用标准 one-to-one 匹配
        loss_dict_raw = criterion(outputs, targets, **metas)
    
    # 计算损失总和
    loss_sum = sum(loss_dict_raw.values())
    
    # 添加 'sup_' 前缀
    loss_dict = {f"sup_{k}": v for k, v in loss_dict_raw.items()}
    
    # 🎯 返回 outputs 以便用于原型更新（包含 encoder_feate）
    return loss_sum, loss_dict, outputs


def compute_unsup_loss(
    student_model: nn.Module,
    teacher_model: nn.Module,
    criterion: nn.Module,
    images_weak: torch.Tensor,
    images_strong: torch.Tensor,
    metas: Dict,
    conf_thresh: float = 0.3,
    scaler=None,
    device: torch.device = None,
    teacher_outputs: Optional[Dict] = None,  # 🔥 新增：预计算的 Teacher 输出
    transform_weak: Optional[Dict] = None,    # 🔥 新增：weak 增强的变换信息
    transform_strong: Optional[Dict] = None, # 🔥 新增：strong 增强的变换信息
    img_size_weak: Optional[Tuple[int, int]] = None,   # 🔥 新增：weak 图像尺寸 (H, W)
    img_size_strong: Optional[Tuple[int, int]] = None, # 🔥 新增：strong 图像尺寸 (H, W)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool, int]:
    """
    计算无监督损失（Semi-DETR 风格）
    
    流程：
    1. Teacher 使用 weak 增强图像生成伪标签
    2. Student 使用 strong 增强图像 + 伪标签计算损失
    
    Args:
        student_model: 学生模型
        teacher_model: 教师模型（EMA）
        criterion: 损失函数
        images_weak: weak 增强图像（用于 Teacher）
        images_strong: strong 增强图像（用于 Student）
        metas: 元信息
        conf_thresh: 伪标签置信度阈值
        scaler: AMP scaler（可选）
        device: 设备
        teacher_outputs: 预计算的 Teacher 输出（可选，避免重复前向传播）
    
    Returns:
        loss_sum: 无监督损失总和（未加权）
        loss_dict: 各项损失字典（带 'unsup_' 前缀）
        has_valid_pseudo: 是否有有效伪标签
        num_pseudo_boxes: 伪标签框数量
    """
    if device is None:
        device = images_weak.device
    
    B_unsup = images_weak.shape[0]
    pseudo_targets = []
    has_valid_pseudo = False
    num_pseudo_boxes = 0
    
    # ========== Step 1: Teacher 生成伪标签 ==========
    # 🔥 如果已经传入 teacher_outputs，直接使用，避免重复前向传播
    if teacher_outputs is None:
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(images_weak, is_teacher=True)
    
    pred_logits = teacher_outputs['pred_logits']  # [B, Q, C]
    pred_boxes_weak = teacher_outputs['pred_boxes']    # [B, Q, 4] (cxcywh 格式，在 weak 坐标系中)
    
    # 🔥 导入变换工具
    from .transform_utils import transform_boxes_cxcywh_weak_to_strong
    
    for b in range(B_unsup):
        probs = pred_logits[b].sigmoid()
        scores, labels = probs.max(dim=-1)
        
        # 置信度过滤
        keep = scores >= conf_thresh
        if keep.sum() > 0:
            has_valid_pseudo = True
            num_pseudo_boxes += keep.sum().item()
            
            # 🔥 获取过滤后的 boxes（在 weak 坐标系中）
            boxes_weak_b = pred_boxes_weak[b][keep].detach()  # [N, 4] (cxcywh)
            
            # 🔥 将 boxes 从 weak 坐标系变换到 strong 坐标系
            if transform_weak is not None and transform_strong is not None and \
               img_size_weak is not None and img_size_strong is not None:
                boxes_strong_b = transform_boxes_cxcywh_weak_to_strong(
                    boxes_weak_b,
                    transform_weak,
                    transform_strong,
                    img_size_weak=img_size_weak,
                    img_size_strong=img_size_strong
                )
            else:
                # 如果没有提供变换信息，直接使用 weak 坐标系的 boxes
                boxes_strong_b = boxes_weak_b
            
            pseudo_targets.append({
                'boxes': boxes_strong_b,  # 变换后的 boxes（在 strong 坐标系中）
                'labels': labels[keep].detach(),
            })
        else:
            # 空目标（保持 batch 大小一致）
            pseudo_targets.append({
                'boxes': torch.zeros((0, 4), device=device),
                'labels': torch.zeros((0,), dtype=torch.long, device=device),
            })
    
    # ========== Step 2: Student 计算无监督损失 ==========
    # 🔥 DDP 关键：无论是否有有效伪标签，都必须执行前向传播
    # 否则不同 GPU 的前向传播次数不一致，会导致梯度同步死锁
    student_model.train()
    
    if scaler is not None:
        with torch.amp.autocast(device_type='cuda', enabled=True):
            outputs = student_model(images_strong, targets=pseudo_targets)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            loss_dict_raw = criterion(outputs, pseudo_targets, **metas)
    else:
        outputs = student_model(images_strong, targets=pseudo_targets)
        loss_dict_raw = criterion(outputs, pseudo_targets, **metas)
    
    # 计算损失总和
    loss_sum = sum(loss_dict_raw.values())
    
    # 🔥 DDP 安全：不管有没有伪标签，都保持损失的梯度流
    # 如果没有伪标签，损失值会很小但不是完全的 0
    # 这样可以避免 DDP 梯度同步问题
    if num_pseudo_boxes == 0:
        # 使用一个很小的系数，保持梯度流但不影响训练
        loss_sum = loss_sum * 1e-8
    
    # 添加 'unsup_' 前缀
    loss_dict = {f"unsup_{k}": v for k, v in loss_dict_raw.items()}
    
    return loss_sum, loss_dict, has_valid_pseudo, num_pseudo_boxes


def weighted_loss(loss_dict: Dict[str, torch.Tensor], weight: float) -> Dict[str, torch.Tensor]:
    """
    对损失字典中的所有值乘以权重（Semi-DETR 风格）
    
    Args:
        loss_dict: 损失字典
        weight: 权重
    
    Returns:
        加权后的损失字典
    """
    return {k: v * weight for k, v in loss_dict.items()}


def compute_unsup_loss_with_consistency(
    student_model: nn.Module,
    teacher_model: nn.Module,
    criterion: nn.Module,
    images_weak: torch.Tensor,
    images_strong: torch.Tensor,
    metas: Dict,
    conf_thresh: float = 0.3,
    consistency_weight: float = 1.0,
    scaler=None,
    device: torch.device = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool, int, Dict[str, float]]:
    """
    计算无监督损失 + 跨视角一致性损失（Semi-DETR 风格）
    
    核心思想：
    1. Teacher 在 weak 增强图像上生成伪标签和特征
    2. Student 在 strong 增强图像上计算伪标签损失
    3. 额外计算 Teacher 和 Student 高置信度查询的特征一致性损失
    
    Args:
        student_model: 学生模型
        teacher_model: 教师模型（EMA）
        criterion: 损失函数
        images_weak: weak 增强图像（用于 Teacher）
        images_strong: strong 增强图像（用于 Student）
        metas: 元信息
        conf_thresh: 伪标签置信度阈值
        consistency_weight: 一致性损失权重
        scaler: AMP scaler（可选）
        device: 设备
    
    Returns:
        loss_sum: 无监督损失总和（伪标签损失 + 一致性损失）
        loss_dict: 各项损失字典（带 'unsup_' 前缀）
        has_valid_pseudo: 是否有有效伪标签
        num_pseudo_boxes: 伪标签框数量
        consistency_stats: 一致性损失统计信息
    """
    if device is None:
        device = images_weak.device
    
    B_unsup = images_weak.shape[0]
    pseudo_targets = []
    has_valid_pseudo = False
    num_pseudo_boxes = 0
    
    # ========== Step 1: Teacher 生成伪标签和特征 ==========
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(images_weak, is_teacher=True)
        teacher_logits = teacher_outputs['pred_logits']  # [B, Q, C]
        teacher_boxes = teacher_outputs['pred_boxes']    # [B, Q, 4]
        
        # 🔥 Teacher 的查询特征会在 Step 3 中获取
        # DEIM 返回 'dec_hs' 而不是 'hs'
        
        for b in range(B_unsup):
            probs = teacher_logits[b].sigmoid()
            scores, labels = probs.max(dim=-1)
            
            # 置信度过滤
            keep = scores >= conf_thresh
            if keep.sum() > 0:
                has_valid_pseudo = True
                num_pseudo_boxes += keep.sum().item()
                pseudo_targets.append({
                    'boxes': teacher_boxes[b][keep].detach(),
                    'labels': labels[keep].detach(),
                    'scores': scores[keep].detach(),  # 保存分数用于一致性筛选
                })
            else:
                pseudo_targets.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'labels': torch.zeros((0,), dtype=torch.long, device=device),
                    'scores': torch.zeros((0,), device=device),
                })
    
    # ========== Step 2: Student 计算无监督损失 ==========
    student_model.train()
    
    if scaler is not None:
        with torch.amp.autocast(device_type='cuda', enabled=True):
            student_outputs = student_model(images_strong, targets=pseudo_targets)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            loss_dict_raw = criterion(student_outputs, pseudo_targets, **metas)
    else:
        student_outputs = student_model(images_strong, targets=pseudo_targets)
        loss_dict_raw = criterion(student_outputs, pseudo_targets, **metas)
    
    # 计算伪标签损失
    loss_pseudo = sum(loss_dict_raw.values())
    
    if not has_valid_pseudo:
        loss_pseudo = loss_pseudo * 0.0
    
    # ========== Step 3: 计算跨视角一致性损失 ==========
    loss_consistency = torch.tensor(0.0, device=device)
    consistency_stats = {'num_consistent_queries': 0, 'consistency_loss': 0.0}
    
    # 获取查询特征（DEIM 返回 'dec_hs' 而不是 'hs'）
    # dec_hs 是一个字典 {layer_idx: [B, Q, C]}
    teacher_query_features = teacher_outputs.get('dec_hs', None)
    student_query_features = student_outputs.get('dec_hs', None)
    student_logits = student_outputs['pred_logits']
    
    if (teacher_query_features is not None and 
        student_query_features is not None and 
        has_valid_pseudo and 
        consistency_weight > 0):
        
        try:
            # 使用最后一层的查询特征
            # dec_hs 是字典 {0: tensor, 1: tensor, ...}
            if isinstance(teacher_query_features, dict):
                last_layer_idx = max(teacher_query_features.keys())
                t_feat = teacher_query_features[last_layer_idx]  # [B, Q, C]
            elif isinstance(teacher_query_features, (list, tuple)):
                t_feat = teacher_query_features[-1]
            else:
                t_feat = teacher_query_features
                
            if isinstance(student_query_features, dict):
                last_layer_idx = max(student_query_features.keys())
                s_feat = student_query_features[last_layer_idx]  # [B, Q, C]
            elif isinstance(student_query_features, (list, tuple)):
                s_feat = student_query_features[-1]
            else:
                s_feat = student_query_features
            
            # 确保形状匹配
            if t_feat.shape != s_feat.shape:
                # 如果形状不匹配（可能由于 DN 查询），跳过一致性损失
                consistency_stats = {'num_consistent_queries': 0, 'consistency_loss': 0.0, 'skipped': 'shape_mismatch'}
            else:
                # 获取分数
                t_scores = teacher_logits.sigmoid().max(dim=-1)[0]  # [B, Q]
                s_scores = student_logits.sigmoid().max(dim=-1)[0]  # [B, Q]
                
                # 只对 Teacher 高置信度的查询计算一致性
                mask = t_scores >= conf_thresh  # [B, Q]
                num_consistent = mask.sum().item()
                
                if num_consistent > 0:
                    # L2 归一化
                    t_feat_norm = F.normalize(t_feat, p=2, dim=-1)
                    s_feat_norm = F.normalize(s_feat, p=2, dim=-1)
                    
                    # 计算 MSE 损失
                    mse_loss = F.mse_loss(s_feat_norm, t_feat_norm.detach(), reduction='none')
                    mse_loss = mse_loss.mean(dim=-1)  # [B, Q]
                    
                    # 应用掩码
                    loss_consistency = (mse_loss * mask.float()).sum() / (mask.sum() + 1e-6)
                    loss_consistency = consistency_weight * loss_consistency
                    
                    consistency_stats = {
                        'num_consistent_queries': num_consistent,
                        'consistency_loss': loss_consistency.item(),
                    }
        except Exception as e:
            # 如果出错，跳过一致性损失但不中断训练
            consistency_stats = {'num_consistent_queries': 0, 'consistency_loss': 0.0, 'error': str(e)}
    
    # ========== Step 4: 合并损失 ==========
    loss_sum = loss_pseudo + loss_consistency
    
    # 添加 'unsup_' 前缀
    loss_dict = {f"unsup_{k}": v for k, v in loss_dict_raw.items()}
    loss_dict['unsup_consistency'] = loss_consistency
    
    return loss_sum, loss_dict, has_valid_pseudo, num_pseudo_boxes, consistency_stats


def compute_unsup_loss_with_roi_consistency(
    student_model: nn.Module,
    teacher_model: nn.Module,
    criterion: nn.Module,
    images_weak: torch.Tensor,
    images_strong: torch.Tensor,
    metas: Dict,
    conf_thresh: float = 0.3,
    consistency_weight: float = 1.0,
    roi_output_size: int = 7,
    scaler=None,
    device: torch.device = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], bool, int, Dict[str, float]]:
    """
    计算无监督损失 + 基于 RoIAlign 的跨视角一致性损失（Semi-DETR 完整版）
    
    核心思想（参考 Semi-DETR）：
    1. Teacher 在 weak 增强图像上生成伪标签
    2. 使用 RoIAlign 从 Teacher 和 Student 的编码器特征中提取伪标签框对应的区域特征
    3. 计算区域特征的一致性损失
    
    优点：
    - 显式对齐目标区域的特征，更精确
    - 不受查询数量和 DN 查询的影响
    
    Args:
        student_model: 学生模型
        teacher_model: 教师模型（EMA）
        criterion: 损失函数
        images_weak: weak 增强图像（用于 Teacher）
        images_strong: strong 增强图像（用于 Student）
        metas: 元信息
        conf_thresh: 伪标签置信度阈值
        consistency_weight: 一致性损失权重
        roi_output_size: RoIAlign 输出尺寸（默认 7x7）
        scaler: AMP scaler（可选）
        device: 设备
    
    Returns:
        loss_sum: 无监督损失总和
        loss_dict: 各项损失字典
        has_valid_pseudo: 是否有有效伪标签
        num_pseudo_boxes: 伪标签框数量
        consistency_stats: 一致性损失统计信息
    """
    from torchvision.ops import roi_align
    
    if device is None:
        device = images_weak.device
    
    B_unsup = images_weak.shape[0]
    pseudo_targets = []
    pseudo_boxes_for_roi = []  # 用于 RoI 提取的框（xyxy 格式，未归一化）
    has_valid_pseudo = False
    num_pseudo_boxes = 0
    
    # ========== Step 1: Teacher 生成伪标签 ==========
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(images_weak, is_teacher=True)
        teacher_logits = teacher_outputs['pred_logits']  # [B, Q, C]
        teacher_boxes = teacher_outputs['pred_boxes']    # [B, Q, 4] (cxcywh, normalized)
        
        # 获取 Teacher 的编码器特征（用于 RoI 一致性）
        teacher_encoder_feats = teacher_outputs.get('encoder_feate', None)
        
        H, W = images_weak.shape[2:]  # 图像尺寸
        
        for b in range(B_unsup):
            probs = teacher_logits[b].sigmoid()
            scores, labels = probs.max(dim=-1)
            
            # 置信度过滤
            keep = scores >= conf_thresh
            if keep.sum() > 0:
                has_valid_pseudo = True
                num_pseudo_boxes += keep.sum().item()
                
                boxes_cxcywh = teacher_boxes[b][keep]  # [N, 4]
                
                # 转换为 xyxy 格式（未归一化，用于 RoI）
                cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
                x1 = (cx - w/2) * W
                y1 = (cy - h/2) * H
                x2 = (cx + w/2) * W
                y2 = (cy + h/2) * H
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # [N, 4]
                
                # 添加 batch 索引
                batch_idx = torch.full((boxes_xyxy.shape[0], 1), b, device=device)
                boxes_with_batch = torch.cat([batch_idx, boxes_xyxy], dim=-1)  # [N, 5]
                pseudo_boxes_for_roi.append(boxes_with_batch)
                
                pseudo_targets.append({
                    'boxes': boxes_cxcywh.detach(),
                    'labels': labels[keep].detach(),
                    'scores': scores[keep].detach(),
                })
            else:
                pseudo_targets.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'labels': torch.zeros((0,), dtype=torch.long, device=device),
                    'scores': torch.zeros((0,), device=device),
                })
    
    # ========== Step 2: Student 计算无监督损失 ==========
    student_model.train()
    
    if scaler is not None:
        with torch.amp.autocast(device_type='cuda', enabled=True):
            student_outputs = student_model(images_strong, targets=pseudo_targets)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            loss_dict_raw = criterion(student_outputs, pseudo_targets, **metas)
    else:
        student_outputs = student_model(images_strong, targets=pseudo_targets)
        loss_dict_raw = criterion(student_outputs, pseudo_targets, **metas)
    
    # 计算伪标签损失
    loss_pseudo = sum(loss_dict_raw.values())
    
    if not has_valid_pseudo:
        loss_pseudo = loss_pseudo * 0.0
    
    # ========== Step 3: 计算基于 RoI 的跨视角一致性损失 ==========
    loss_consistency = torch.tensor(0.0, device=device)
    consistency_stats = {'num_consistent_queries': 0, 'consistency_loss': 0.0}
    
    # 获取 Student 的编码器特征
    student_encoder_feats = student_outputs.get('encoder_feate', None)
    
    if (teacher_encoder_feats is not None and 
        student_encoder_feats is not None and 
        has_valid_pseudo and 
        len(pseudo_boxes_for_roi) > 0 and
        consistency_weight > 0):
        
        try:
            # 合并所有伪标签框
            all_rois = torch.cat(pseudo_boxes_for_roi, dim=0)  # [total_boxes, 5]
            num_rois = all_rois.shape[0]
            
            if num_rois > 0:
                # 选择合适尺度的特征图（通常是最大的那个）
                # encoder_feate 是一个列表 [feat1, feat2, feat3]
                if isinstance(teacher_encoder_feats, (list, tuple)):
                    # 使用第一个（最大分辨率）特征图
                    t_feat_map = teacher_encoder_feats[0]  # [B, C, H, W]
                    s_feat_map = student_encoder_feats[0]
                else:
                    t_feat_map = teacher_encoder_feats
                    s_feat_map = student_encoder_feats
                
                # 计算特征图相对于输入图像的缩放比例
                feat_H, feat_W = t_feat_map.shape[2:]
                spatial_scale = feat_H / H  # 假设 H 和 W 缩放比例相同
                
                # 使用 RoIAlign 提取区域特征
                with torch.no_grad():
                    t_roi_feats = roi_align(
                        t_feat_map.float(),
                        all_rois,
                        output_size=roi_output_size,
                        spatial_scale=spatial_scale,
                        aligned=True
                    )  # [num_rois, C, roi_size, roi_size]
                
                s_roi_feats = roi_align(
                    s_feat_map.float(),
                    all_rois,
                    output_size=roi_output_size,
                    spatial_scale=spatial_scale,
                    aligned=True
                )  # [num_rois, C, roi_size, roi_size]
                
                # 全局平均池化
                t_roi_pooled = t_roi_feats.mean(dim=[2, 3])  # [num_rois, C]
                s_roi_pooled = s_roi_feats.mean(dim=[2, 3])  # [num_rois, C]
                
                # L2 归一化
                t_roi_norm = F.normalize(t_roi_pooled, p=2, dim=-1)
                s_roi_norm = F.normalize(s_roi_pooled, p=2, dim=-1)
                
                # 计算 MSE 损失
                loss_consistency = F.mse_loss(s_roi_norm, t_roi_norm.detach())
                loss_consistency = consistency_weight * loss_consistency
                
                consistency_stats = {
                    'num_consistent_queries': num_rois,
                    'consistency_loss': loss_consistency.item(),
                    'method': 'roi_align',
                }
                
        except Exception as e:
            # 如果出错，跳过一致性损失但不中断训练
            consistency_stats = {'num_consistent_queries': 0, 'consistency_loss': 0.0, 'error': str(e)}
    
    # ========== Step 4: 合并损失 ==========
    loss_sum = loss_pseudo + loss_consistency
    
    # 添加 'unsup_' 前缀
    loss_dict = {f"unsup_{k}": v for k, v in loss_dict_raw.items()}
    loss_dict['unsup_consistency'] = loss_consistency
    
    return loss_sum, loss_dict, has_valid_pseudo, num_pseudo_boxes, consistency_stats
