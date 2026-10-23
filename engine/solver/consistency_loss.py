"""
Semi-DETR 风格的跨视角一致性损失

核心思想：
1. Teacher 在 weak 增强图像上生成特征
2. Student 在 strong 增强图像上生成特征
3. 通过一致性损失约束两者的特征表示相似

这有助于：
- 提高伪标签的质量筛选
- 让 Student 学习到增强不变的特征表示
- 防止 Student 过拟合到 strong 增强的噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ConsistencyLoss(nn.Module):
    """
    跨视角一致性损失（简化版 Semi-DETR）
    
    计算 Teacher (weak) 和 Student (strong) 特征之间的一致性损失
    """
    
    def __init__(self, 
                 loss_weight: float = 1.0,
                 temperature: float = 0.5,
                 use_l2_norm: bool = True):
        """
        Args:
            loss_weight: 一致性损失的权重
            temperature: 温度参数（用于 softmax）
            use_l2_norm: 是否对特征进行 L2 归一化
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.use_l2_norm = use_l2_norm
    
    def forward(self,
                teacher_features: torch.Tensor,
                student_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算一致性损失
        
        Args:
            teacher_features: Teacher 特征 [B, N, C] 或 [B, C, H, W]
            student_features: Student 特征 [B, N, C] 或 [B, C, H, W]
            mask: 可选的掩码，指示哪些位置参与损失计算
        
        Returns:
            一致性损失
        """
        # 确保形状一致
        if teacher_features.shape != student_features.shape:
            raise ValueError(f"Shape mismatch: teacher {teacher_features.shape} vs student {student_features.shape}")
        
        # L2 归一化
        if self.use_l2_norm:
            teacher_features = F.normalize(teacher_features, p=2, dim=-1)
            student_features = F.normalize(student_features, p=2, dim=-1)
        
        # 计算 MSE 损失
        loss = F.mse_loss(student_features, teacher_features.detach(), reduction='none')
        
        # 应用掩码
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return self.loss_weight * loss


def compute_box_consistency_loss(
    teacher_boxes: List[torch.Tensor],
    student_boxes: List[torch.Tensor],
    teacher_scores: List[torch.Tensor],
    student_scores: List[torch.Tensor],
    score_thresh: float = 0.3,
    iou_thresh: float = 0.5,
) -> Tuple[torch.Tensor, int]:
    """
    计算框级别的一致性损失
    
    只对 Teacher 和 Student 都检测到的目标计算一致性损失
    
    Args:
        teacher_boxes: Teacher 预测框列表 [B, N, 4]
        student_boxes: Student 预测框列表 [B, M, 4]
        teacher_scores: Teacher 预测分数列表 [B, N]
        student_scores: Student 预测分数列表 [B, M]
        score_thresh: 分数阈值
        iou_thresh: IoU 阈值，用于匹配 Teacher 和 Student 的预测
    
    Returns:
        一致性损失, 匹配的框数量
    """
    from engine.deim.box_ops import box_iou
    
    total_loss = 0.0
    num_matches = 0
    device = teacher_boxes[0].device if len(teacher_boxes) > 0 and teacher_boxes[0].numel() > 0 else 'cuda'
    
    for t_boxes, s_boxes, t_scores, s_scores in zip(
        teacher_boxes, student_boxes, teacher_scores, student_scores
    ):
        # 过滤低分预测
        t_mask = t_scores >= score_thresh
        s_mask = s_scores >= score_thresh
        
        t_boxes_valid = t_boxes[t_mask]
        s_boxes_valid = s_boxes[s_mask]
        t_scores_valid = t_scores[t_mask]
        s_scores_valid = s_scores[s_mask]
        
        if t_boxes_valid.numel() == 0 or s_boxes_valid.numel() == 0:
            continue
        
        # 计算 IoU 矩阵
        ious = box_iou(t_boxes_valid, s_boxes_valid)
        
        # 对于每个 Teacher 框，找到最匹配的 Student 框
        max_ious, matched_idx = ious.max(dim=1)
        
        # 只保留 IoU > thresh 的匹配
        valid_matches = max_ious >= iou_thresh
        
        if valid_matches.sum() == 0:
            continue
        
        # 计算匹配框的分数一致性损失
        matched_t_scores = t_scores_valid[valid_matches]
        matched_s_scores = s_scores_valid[matched_idx[valid_matches]]
        
        # L1 损失
        score_loss = F.l1_loss(matched_s_scores, matched_t_scores.detach())
        
        # 计算匹配框的位置一致性损失
        matched_t_boxes = t_boxes_valid[valid_matches]
        matched_s_boxes = s_boxes_valid[matched_idx[valid_matches]]
        
        box_loss = F.l1_loss(matched_s_boxes, matched_t_boxes.detach())
        
        total_loss = total_loss + score_loss + box_loss
        num_matches += valid_matches.sum().item()
    
    if num_matches == 0:
        return torch.tensor(0.0, device=device), 0
    
    return total_loss / max(num_matches, 1), num_matches


def compute_query_consistency_loss(
    teacher_query_features: torch.Tensor,
    student_query_features: torch.Tensor,
    teacher_scores: torch.Tensor,
    student_scores: torch.Tensor,
    score_thresh: float = 0.3,
    loss_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算查询级别的一致性损失（Semi-DETR 核心）
    
    对于高置信度的 Teacher 查询，约束 Student 的对应查询特征与其一致
    
    Args:
        teacher_query_features: Teacher 查询特征 [B, Q, C]
        student_query_features: Student 查询特征 [B, Q, C]
        teacher_scores: Teacher 预测分数 [B, Q, num_classes]
        student_scores: Student 预测分数 [B, Q, num_classes]
        score_thresh: 分数阈值
        loss_weight: 损失权重
    
    Returns:
        一致性损失, 统计信息
    """
    B, Q, C = teacher_query_features.shape
    device = teacher_query_features.device
    
    # 获取最大分数
    t_max_scores = teacher_scores.sigmoid().max(dim=-1)[0]  # [B, Q]
    s_max_scores = student_scores.sigmoid().max(dim=-1)[0]  # [B, Q]
    
    # 创建掩码：只对 Teacher 高置信度的查询计算一致性
    mask = t_max_scores >= score_thresh  # [B, Q]
    
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device), {'num_consistent_queries': 0}
    
    # L2 归一化
    t_feat_norm = F.normalize(teacher_query_features, p=2, dim=-1)
    s_feat_norm = F.normalize(student_query_features, p=2, dim=-1)
    
    # 计算 MSE 损失（只对高置信度查询）
    mse_loss = F.mse_loss(s_feat_norm, t_feat_norm.detach(), reduction='none')  # [B, Q, C]
    mse_loss = mse_loss.mean(dim=-1)  # [B, Q]
    
    # 应用掩码
    masked_loss = (mse_loss * mask.float()).sum() / (mask.sum() + 1e-6)
    
    stats = {
        'num_consistent_queries': mask.sum().item(),
        'avg_teacher_score': t_max_scores[mask].mean().item() if mask.sum() > 0 else 0,
    }
    
    return loss_weight * masked_loss, stats


def compute_feature_consistency_loss(
    teacher_features: List[torch.Tensor],
    student_features: List[torch.Tensor],
    pseudo_boxes: List[torch.Tensor],
    loss_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算特征级别的一致性损失
    
    在伪标签框对应的区域计算 Teacher 和 Student 特征的一致性
    
    Args:
        teacher_features: Teacher 多尺度特征 [level][B, C, H, W]
        student_features: Student 多尺度特征 [level][B, C, H, W]
        pseudo_boxes: 伪标签框列表 [B][N, 4]，归一化的 cxcywh 格式
        loss_weight: 损失权重
    
    Returns:
        一致性损失, 统计信息
    """
    device = teacher_features[0].device
    total_loss = torch.tensor(0.0, device=device)
    num_boxes = 0
    
    # 对每个尺度计算一致性
    for level, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
        B, C, H, W = t_feat.shape
        
        for b in range(B):
            boxes = pseudo_boxes[b]  # [N, 4]
            if boxes.numel() == 0:
                continue
            
            # 将归一化框转换为特征图坐标
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = ((cx - w/2) * W).long().clamp(0, W-1)
            y1 = ((cy - h/2) * H).long().clamp(0, H-1)
            x2 = ((cx + w/2) * W).long().clamp(0, W-1)
            y2 = ((cy + h/2) * H).long().clamp(0, H-1)
            
            for i in range(len(boxes)):
                if x2[i] <= x1[i] or y2[i] <= y1[i]:
                    continue
                
                # 提取区域特征
                t_roi = t_feat[b, :, y1[i]:y2[i]+1, x1[i]:x2[i]+1]
                s_roi = s_feat[b, :, y1[i]:y2[i]+1, x1[i]:x2[i]+1]
                
                # 全局平均池化
                t_roi_pooled = t_roi.mean(dim=[1, 2])  # [C]
                s_roi_pooled = s_roi.mean(dim=[1, 2])  # [C]
                
                # L2 归一化后计算 MSE
                t_roi_norm = F.normalize(t_roi_pooled, p=2, dim=0)
                s_roi_norm = F.normalize(s_roi_pooled, p=2, dim=0)
                
                loss = F.mse_loss(s_roi_norm, t_roi_norm.detach())
                total_loss = total_loss + loss
                num_boxes += 1
    
    if num_boxes == 0:
        return torch.tensor(0.0, device=device), {'num_roi_boxes': 0}
    
    avg_loss = total_loss / num_boxes
    
    return loss_weight * avg_loss, {'num_roi_boxes': num_boxes}

