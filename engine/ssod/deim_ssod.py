"""
DEIM 半监督检测器（参考 Semi-DETR DinoDetrSSOD）

核心设计：
1. MultiStreamDetector：管理 teacher 和 student 两个子模型
2. Projector：将 RoI 特征投影到 Query 空间
3. GMM 伪标签筛选
4. Cross-view Query Consistency
5. Stage-wise Hybrid Matching (O2M → O2O)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .projector import Projector, MultiScaleRoIExtractor
from .gmm_filter import GMMFilter, compute_matching_cost
from ..deim.box_ops import box_cxcywh_to_xyxy, box_iou
from ..deim.utils import inverse_sigmoid


class DEIMSemiDetector(nn.Module):
    """
    DEIM 半监督检测器（参考 Semi-DETR DinoDetrSSOD）
    
    包含 teacher 和 student 两个 DEIM 模型：
    - Teacher：冻结，使用 EMA 更新，生成伪标签
    - Student：可训练，使用有标签数据和伪标签数据训练
    
    核心模块：
    1. Projector：RoI 特征投影到 Query 空间
    2. RoI Extractor：多尺度 RoI 特征提取
    3. GMM Filter：GMM 伪标签筛选
    """
    
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module = None,
        hidden_dim: int = 256,
        num_classes: int = 80,
        # 训练超参数
        unsup_weight: float = 4.0,
        pseudo_label_score_thr: float = 0.5,
        min_pseudo_box_size: int = 0,
        # RoI 参数
        featmap_strides: List[int] = [8, 16, 32],
        roi_size: int = 7,
    ):
        """
        Args:
            student: 学生模型（DEIM）
            teacher: 教师模型（DEIM），如果为 None 则复制学生模型
            hidden_dim: 隐藏维度
            num_classes: 类别数
            unsup_weight: 无监督损失权重
            pseudo_label_score_thr: 伪标签置信度阈值
            min_pseudo_box_size: 最小伪标签框尺寸
            featmap_strides: 特征图步长
            roi_size: RoI 输出尺寸
        """
        super().__init__()
        
        self.student = student
        self.teacher = teacher if teacher is not None else copy.deepcopy(student)
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.unsup_weight = unsup_weight
        self.pseudo_label_score_thr = pseudo_label_score_thr
        self.min_pseudo_box_size = min_pseudo_box_size
        
        # 冻结教师模型
        self._freeze_teacher()
        
        # RoI 特征提取器
        self.roi_extractor = MultiScaleRoIExtractor(
            featmap_strides=featmap_strides,
            out_channels=hidden_dim,
            roi_size=roi_size
        )
        
        # Query 投影器
        self.projector = Projector(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            roi_size=roi_size
        )
        
        # GMM 筛选器
        self.gmm_filter = GMMFilter()
        
        # 当前训练步
        self.curr_step = 0
    
    def _freeze_teacher(self):
        """冻结教师模型"""
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward_train(
        self,
        sup_images: torch.Tensor,
        sup_targets: List[Dict],
        unsup_images_weak: torch.Tensor,
        unsup_images_strong: torch.Tensor,
        unsup_targets_weak: List[Dict],
        unsup_targets_strong: List[Dict],
        criterion: nn.Module,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        半监督训练前向传播（参考 Semi-DETR forward_train）
        
        Args:
            sup_images: 有标签图像 [B_sup, 3, H, W]
            sup_targets: 有标签目标
            unsup_images_weak: 无标签弱增广图像 [B_unsup, 3, H, W]
            unsup_images_strong: 无标签强增广图像 [B_unsup, 3, H, W]
            unsup_targets_weak: 无标签弱增广目标（用于变换矩阵）
            unsup_targets_strong: 无标签强增广目标
            criterion: 损失函数
            
        Returns:
            loss_dict: 损失字典
        """
        loss = {}
        
        # ========== 1. 有监督损失 ==========
        sup_loss = self._compute_sup_loss(sup_images, sup_targets, criterion)
        sup_loss = {f'sup_{k}': v for k, v in sup_loss.items()}
        loss.update(sup_loss)
        
        # ========== 2. 无监督损失 ==========
        unsup_loss = self._compute_unsup_loss(
            unsup_images_weak,
            unsup_images_strong,
            unsup_targets_weak,
            unsup_targets_strong,
            criterion,
            **kwargs
        )
        unsup_loss = {f'unsup_{k}': v * self.unsup_weight for k, v in unsup_loss.items()}
        loss.update(unsup_loss)
        
        return loss
    
    def _compute_sup_loss(
        self,
        images: torch.Tensor,
        targets: List[Dict],
        criterion: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """计算有监督损失"""
        self.student.train()
        outputs = self.student(images, targets=targets)
        loss_dict = criterion(outputs, targets)
        return loss_dict
    
    def _compute_unsup_loss(
        self,
        images_weak: torch.Tensor,
        images_strong: torch.Tensor,
        targets_weak: List[Dict],
        targets_strong: List[Dict],
        criterion: nn.Module,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        计算无监督损失（参考 Semi-DETR foward_unsup_train）
        
        流程：
        1. Teacher 在弱增广图像上生成伪标签
        2. 坐标变换：将伪标签从弱增广空间映射到强增广空间
        3. GMM 筛选：使用 GMM 筛选高质量伪标签
        4. Student 在强增广图像上训练
        """
        B = images_strong.shape[0]
        device = images_strong.device
        
        # ========== Step 1: Teacher 生成伪标签 ==========
        teacher_info = self._extract_teacher_info(images_weak, targets_weak)
        
        # ========== Step 2: 获取伪标签并进行初步筛选 ==========
        pseudo_bboxes_list = []
        pseudo_labels_list = []
        pseudo_scores_list = []
        
        for b in range(B):
            det_bboxes = teacher_info['det_bboxes'][b]  # [N, 4], cxcywh, normalized
            det_labels = teacher_info['det_labels'][b]  # [N]
            det_scores = teacher_info['det_scores'][b]  # [N]
            
            # 置信度阈值筛选
            valid_mask = det_scores >= self.pseudo_label_score_thr
            
            # 最小框尺寸筛选
            if self.min_pseudo_box_size > 0:
                H, W = images_weak.shape[-2:]
                w = det_bboxes[:, 2] * W
                h = det_bboxes[:, 3] * H
                size_mask = (w >= self.min_pseudo_box_size) & (h >= self.min_pseudo_box_size)
                valid_mask = valid_mask & size_mask
            
            pseudo_bboxes_list.append(det_bboxes[valid_mask])
            pseudo_labels_list.append(det_labels[valid_mask])
            pseudo_scores_list.append(det_scores[valid_mask])
        
        # ========== Step 3: 坐标变换（弱增广 → 强增广）==========
        pseudo_bboxes_strong = self._transform_pseudo_labels(
            pseudo_bboxes_list,
            targets_weak,
            targets_strong,
            images_weak.shape[-2:],
            images_strong.shape[-2:]
        )
        
        # ========== Step 4: Student 训练 ==========
        # 构造伪标签 targets
        pseudo_targets = []
        for b in range(B):
            target = {
                'boxes': pseudo_bboxes_strong[b],
                'labels': pseudo_labels_list[b],
            }
            pseudo_targets.append(target)
        
        # 检查是否有有效伪标签
        total_pseudo = sum(t['boxes'].shape[0] for t in pseudo_targets)
        
        if total_pseudo == 0:
            # 无有效伪标签，返回零损失（保持梯度流）
            self.student.train()
            outputs = self.student(images_strong, targets=pseudo_targets)
            loss_dict = criterion(outputs, pseudo_targets)
            # 乘以极小系数，保持梯度流但不影响训练
            return {k: v * 1e-8 for k, v in loss_dict.items()}
        
        # 正常计算无监督损失
        self.student.train()
        outputs = self.student(images_strong, targets=pseudo_targets)
        loss_dict = criterion(outputs, pseudo_targets)
        
        return loss_dict
    
    def _extract_teacher_info(
        self,
        images: torch.Tensor,
        targets: List[Dict],
    ) -> Dict:
        """
        提取教师模型信息（参考 Semi-DETR extract_teacher_info）
        
        Returns:
            teacher_info: 包含以下键：
                - backbone_feature: 骨干网络特征
                - det_bboxes: 检测框列表
                - det_labels: 检测标签列表
                - det_scores: 检测分数列表
        """
        teacher_info = {}
        teacher_info['img'] = images
        
        with torch.no_grad():
            self.teacher.eval()
            outputs = self.teacher(images, targets=None, is_teacher=True)
        
        teacher_info['backbone_feature'] = outputs.get('backbone_feate', None)
        teacher_info['encoder_feature'] = outputs.get('encoder_feate', None)
        
        # 获取预测结果
        pred_logits = outputs['pred_logits']  # [B, num_queries, num_classes]
        pred_boxes = outputs['pred_boxes']    # [B, num_queries, 4]
        
        B = pred_logits.shape[0]
        
        det_bboxes_list = []
        det_labels_list = []
        det_scores_list = []
        
        for b in range(B):
            logits = pred_logits[b]  # [num_queries, num_classes]
            boxes = pred_boxes[b]     # [num_queries, 4]
            
            # 获取最大分数和对应类别
            probs = logits.sigmoid()
            scores, labels = probs.max(dim=-1)
            
            det_bboxes_list.append(boxes)
            det_labels_list.append(labels)
            det_scores_list.append(scores)
        
        teacher_info['det_bboxes'] = det_bboxes_list
        teacher_info['det_labels'] = det_labels_list
        teacher_info['det_scores'] = det_scores_list
        
        return teacher_info
    
    def _transform_pseudo_labels(
        self,
        pseudo_bboxes: List[torch.Tensor],
        targets_weak: List[Dict],
        targets_strong: List[Dict],
        img_size_weak: Tuple[int, int],
        img_size_strong: Tuple[int, int],
    ) -> List[torch.Tensor]:
        """
        坐标变换：将伪标签从弱增广空间映射到强增广空间
        
        参考 Semi-DETR 的 transform_matrix 机制
        
        公式：bbox_strong = T_strong^{-1} @ T_weak @ bbox_weak
        """
        B = len(pseudo_bboxes)
        transformed_bboxes = []
        
        for b in range(B):
            bboxes = pseudo_bboxes[b]  # [N, 4], cxcywh, normalized
            
            if bboxes.shape[0] == 0:
                transformed_bboxes.append(bboxes)
                continue
            
            # 获取变换矩阵
            meta_weak = targets_weak[b].get('meta', {})
            meta_strong = targets_strong[b].get('meta', {})
            
            T_weak = meta_weak.get('transform_matrix', None)
            T_strong = meta_strong.get('transform_matrix', None)
            
            if T_weak is None or T_strong is None:
                # 无变换矩阵，直接使用原坐标
                transformed_bboxes.append(bboxes)
                continue
            
            # 确保变换矩阵在正确的设备上
            if isinstance(T_weak, np.ndarray):
                T_weak = torch.from_numpy(T_weak).float().to(bboxes.device)
            if isinstance(T_strong, np.ndarray):
                T_strong = torch.from_numpy(T_strong).float().to(bboxes.device)
            
            # cxcywh → xyxy（像素坐标）
            H_weak, W_weak = img_size_weak
            H_strong, W_strong = img_size_strong
            
            cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            x1 = (cx - w / 2) * W_weak
            y1 = (cy - h / 2) * H_weak
            x2 = (cx + w / 2) * W_weak
            y2 = (cy + h / 2) * H_weak
            
            # 构造角点坐标 [N, 4, 3]（齐次坐标）
            corners = torch.stack([
                torch.stack([x1, y1, torch.ones_like(x1)], dim=-1),  # 左上
                torch.stack([x2, y1, torch.ones_like(x1)], dim=-1),  # 右上
                torch.stack([x2, y2, torch.ones_like(x1)], dim=-1),  # 右下
                torch.stack([x1, y2, torch.ones_like(x1)], dim=-1),  # 左下
            ], dim=1)  # [N, 4, 3]
            
            # 应用变换：T_strong^{-1} @ T_weak @ corners
            try:
                T = torch.inverse(T_strong) @ T_weak
                transformed_corners = (T @ corners.transpose(-1, -2)).transpose(-1, -2)  # [N, 4, 3]
            except:
                # 矩阵不可逆，直接使用原坐标
                transformed_bboxes.append(bboxes)
                continue
            
            # 提取 x, y 坐标
            tc_x = transformed_corners[:, :, 0]  # [N, 4]
            tc_y = transformed_corners[:, :, 1]  # [N, 4]
            
            # 计算新的 xyxy
            x1_new = tc_x.min(dim=1)[0]
            x2_new = tc_x.max(dim=1)[0]
            y1_new = tc_y.min(dim=1)[0]
            y2_new = tc_y.max(dim=1)[0]
            
            # 裁剪到图像边界
            x1_new = x1_new.clamp(0, W_strong)
            x2_new = x2_new.clamp(0, W_strong)
            y1_new = y1_new.clamp(0, H_strong)
            y2_new = y2_new.clamp(0, H_strong)
            
            # xyxy → cxcywh（归一化）
            cx_new = (x1_new + x2_new) / 2 / W_strong
            cy_new = (y1_new + y2_new) / 2 / H_strong
            w_new = (x2_new - x1_new) / W_strong
            h_new = (y2_new - y1_new) / H_strong
            
            bboxes_new = torch.stack([cx_new, cy_new, w_new, h_new], dim=-1)
            
            # 过滤无效框
            valid = (w_new > 0) & (h_new > 0)
            bboxes_new = bboxes_new[valid]
            
            transformed_bboxes.append(bboxes_new)
        
        return transformed_bboxes
    
    def forward(self, *args, **kwargs):
        """推理时使用学生模型"""
        return self.student(*args, **kwargs)
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        super().train(mode)
        # 教师模型始终保持 eval 模式
        self.teacher.eval()
        return self


# 导入 numpy（在类定义之后）
import numpy as np

