"""
对比学习模块：用于半监督目标检测中区分目标物种和非目标植物

核心思想：
1. 正样本：GT标注的目标杂草特征
2. 负样本：背景区域、其他植物（非目标）的特征
3. 通过对比损失让模型学会区分"目标 vs 非目标"

使用场景：
- 解决模型将玉米叶子误识别为目标杂草的问题
- 增强语义原型的区分能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import random
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def compute_vdvi_from_rgb(rgb_tensor: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406], 
                           std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    从归一化的 RGB 张量实时计算 VDVI
    
    VDVI = (2G - R - B) / (2G + R + B)
    
    Args:
        rgb_tensor: [B, 3, H, W] 归一化后的 RGB 张量
        mean: ImageNet 均值（用于反归一化）
        std: ImageNet 标准差（用于反归一化）
    
    Returns:
        vdvi: [B, 1, H, W] VDVI 图，值域 [-1, 1]
    """
    device = rgb_tensor.device
    
    # 反归一化：rgb = rgb * std + mean
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    rgb_denorm = rgb_tensor * std_t + mean_t  # [B, 3, H, W], 值域 [0, 1]
    
    # 提取 R, G, B 通道
    R = rgb_denorm[:, 0:1, :, :]  # [B, 1, H, W]
    G = rgb_denorm[:, 1:2, :, :]
    B = rgb_denorm[:, 2:3, :, :]
    
    # 计算 VDVI
    numerator = 2.0 * G - R - B
    denominator = 2.0 * G + R + B
    
    # 避免除零
    denominator = torch.where(denominator == 0, torch.ones_like(denominator) * 1e-10, denominator)
    
    vdvi = numerator / denominator  # [B, 1, H, W]
    
    return vdvi


class ContrastiveLoss(nn.Module):
    """
    对比损失（InfoNCE / NT-Xent 风格）
    
    L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    
    其中：
    - z_i: anchor 特征
    - z_j: 正样本特征
    - z_k: 所有样本（正样本 + 负样本）
    - τ: 温度参数
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,      # [N, D] anchor 特征
        positive: torch.Tensor,    # [N, D] 正样本特征
        negatives: torch.Tensor,   # [M, D] 负样本特征
    ) -> torch.Tensor:
        """
        计算 InfoNCE 损失
        
        Args:
            anchor: anchor 特征 [N, D]
            positive: 正样本特征 [N, D]（与 anchor 一一对应）
            negatives: 负样本特征 [M, D]（所有 anchor 共享）
        
        Returns:
            loss: 标量损失
        """
        if anchor.shape[0] == 0:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)
        
        # L2 归一化
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # 计算正样本相似度 [N]
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature
        
        # 计算负样本相似度 [N, M]
        neg_sim = torch.mm(anchor, negatives.t()) / self.temperature
        
        # InfoNCE: -log(exp(pos) / (exp(pos) + Σ exp(neg)))
        # = -pos + log(exp(pos) + Σ exp(neg))
        # = -pos + logsumexp([pos, neg1, neg2, ...])
        
        # 合并正负样本 [N, 1+M]
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # 计算损失
        loss = -pos_sim + torch.logsumexp(all_sim, dim=1)
        
        return loss.mean()


class PrototypeContrastiveLoss(nn.Module):
    """
    基于语义原型的对比损失
    
    思想：
    - 目标样本的特征应该靠近原型
    - 非目标样本的特征应该远离原型
    
    适用于当前场景：让模型区分"目标杂草"和"玉米叶子"
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin  # 负样本与原型的最小距离
    
    def forward(
        self,
        prototype: torch.Tensor,      # [D] 语义原型
        positive_feats: torch.Tensor, # [N, D] 目标样本特征
        negative_feats: torch.Tensor, # [M, D] 非目标样本特征
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算原型对比损失
        
        Args:
            prototype: 语义原型向量 [D]
            positive_feats: 目标样本（GT框）提取的特征 [N, D]
            negative_feats: 非目标样本（背景/其他植物）提取的特征 [M, D]
        
        Returns:
            loss: 标量损失
            stats: 统计信息
        """
        stats = {}
        device = prototype.device
        
        # L2 归一化
        prototype = F.normalize(prototype.unsqueeze(0), dim=-1)  # [1, D]
        
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ========== 正样本损失：拉近目标样本与原型 ==========
        if positive_feats.shape[0] > 0:
            positive_feats = F.normalize(positive_feats, dim=-1)  # [N, D]
            pos_sim = torch.mm(positive_feats, prototype.t()).squeeze(-1)  # [N]
            
            # 正样本损失：最大化相似度（即最小化 1 - sim）
            pos_loss = (1 - pos_sim).mean()
            loss = loss + pos_loss
            
            stats['pos_sim_mean'] = pos_sim.mean().item()
            stats['pos_sim_min'] = pos_sim.min().item()
        
        # ========== 负样本损失：推远非目标样本与原型 ==========
        if negative_feats.shape[0] > 0:
            negative_feats = F.normalize(negative_feats, dim=-1)  # [M, D]
            neg_sim = torch.mm(negative_feats, prototype.t()).squeeze(-1)  # [M]
            
            # 负样本损失：惩罚相似度高于 margin 的负样本
            # hinge loss: max(0, sim - margin)
            neg_loss = F.relu(neg_sim - self.margin).mean()
            loss = loss + neg_loss
            
            stats['neg_sim_mean'] = neg_sim.mean().item()
            stats['neg_sim_max'] = neg_sim.max().item()
            stats['neg_violated'] = (neg_sim > self.margin).sum().item()
        
        stats['total_loss'] = loss.item()
        
        return loss, stats


class NegativeSampleMiner:
    """
    负样本挖掘器：从图像中提取非目标区域的特征作为负样本
    
    策略（改进版）：
    1. 🔥 VDVI 硬负样本：高 VDVI 但不是目标的区域（玉米叶子）← 最重要！
    2. 随机背景采样：从没有目标框的区域随机采样（土壤等）
    3. 低置信度样本：被置信度筛选过滤掉的预测框（备用）
    """
    
    def __init__(
        self,
        num_neg_samples: int = 16,
        min_neg_size: float = 0.02,
        max_neg_size: float = 0.15,
        iou_thresh: float = 0.1,  # 🔥 与正样本的最大 IoU（降低到 0.1，更严格排除目标区域）
    ):
        self.num_neg_samples = num_neg_samples
        self.min_neg_size = min_neg_size
        self.max_neg_size = max_neg_size
        self.iou_thresh = iou_thresh
    
    def mine_vdvi_hard_negatives(
        self,
        encoder_feats: List[torch.Tensor],
        vdvi_map: torch.Tensor,           # [H, W] VDVI 图
        positive_boxes: torch.Tensor,      # [N, 4] GT 框 cxcywh normalized
        batch_id: int,
        img_size: Tuple[int, int],
        roi_extractor,
        vdvi_thresh: float = 0.03,         # VDVI 阈值（高于此值是绿色植物）
        num_samples: int = 4,              # 🔥 每张图的负样本数量（默认 4）
        return_boxes: bool = False,        # 🔥 是否返回框的位置（用于可视化）
        pred_boxes: Optional[torch.Tensor] = None,   # 🔥 预测框 [Q, 4]
        pred_scores: Optional[torch.Tensor] = None,  # 🔥 预测分数 [Q] 用于排序
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        🔥 从"高 VDVI 但不是目标"的区域挖掘硬负样本（玉米叶子）
        
        流程：
        1. 从 300 个预测框中筛选
        2. VDVI 筛选：保留高 VDVI 区域的框（绿色植物）
        3. 排除正样本：去除与 GT 框重叠的预测
        4. 按分数排序：从高到低排序，选取 top-k 作为硬负样本
        
        思想：
        - 高分数 + 高 VDVI + 非目标 = 最难的负样本（玉米叶子被误认为目标）
        
        Args:
            vdvi_map: VDVI 图 [H, W]
            positive_boxes: GT 框（避免采样到目标区域）
            pred_boxes: 预测框 [Q, 4]
            pred_scores: 预测分数 [Q]（用于排序，选取高分预测作为硬负样本）
            num_samples: 每张图采样的负样本数量（默认 4）
            
        Returns:
            neg_feats: 硬负样本特征 [M, D]
        """
        device = encoder_feats[0].device
        H, W = img_size
        
        if vdvi_map is None:
            if return_boxes:
                return torch.zeros((0, encoder_feats[0].shape[1]), device=device), torch.zeros((0, 4), device=device)
            return torch.zeros((0, encoder_feats[0].shape[1]), device=device)
        
        # 确保 VDVI 在正确设备上
        vdvi_map = vdvi_map.to(device)
        
        # 找到高 VDVI 区域（绿色植物）
        high_vdvi_mask = vdvi_map > vdvi_thresh
        
        # 排除 GT 框区域
        if positive_boxes.shape[0] > 0:
            # 将 GT 框转换为像素坐标
            for box in positive_boxes:
                cx, cy, w, h = box
                x1 = int(max(0, (cx - w/2) * W))
                y1 = int(max(0, (cy - h/2) * H))
                x2 = int(min(W, (cx + w/2) * W))
                y2 = int(min(H, (cy + h/2) * H))
                # 扩大排除区域（1.5 倍）避免边缘效应
                pad_w = int((x2 - x1) * 0.25)
                pad_h = int((y2 - y1) * 0.25)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(W, x2 + pad_w)
                y2 = min(H, y2 + pad_h)
                high_vdvi_mask[y1:y2, x1:x2] = False
        
        # 🔥 向量化实现：避免逐框循环，提高效率，防止训练卡死
        # 流程：VDVI 筛选 → 排除正样本 → 按分数排序 → 选 top-k
        
        neg_boxes = torch.zeros((0, 4), device=device)
        
        if pred_boxes is not None and pred_scores is not None and pred_scores.numel() > 0:
            num_preds = pred_boxes.shape[0]
            
            # 🔥 批量计算所有框的中心像素位置（向量化）
            cx_all = pred_boxes[:, 0]  # [Q]
            cy_all = pred_boxes[:, 1]  # [Q]
            px_all = (cx_all * W).long().clamp(0, W - 1)  # [Q]
            py_all = (cy_all * H).long().clamp(0, H - 1)  # [Q]
            
            # 🔥 条件1：批量检查 VDVI（向量化索引）
            vdvi_mask = high_vdvi_mask[py_all, px_all]  # [Q] bool
            
            # 🔥 条件2：批量计算与 GT 的 IoU（避免循环内计算）
            if positive_boxes.shape[0] > 0:
                # 一次性计算所有预测框与所有 GT 框的 IoU [Q, N_gt]
                all_iou = self._box_iou_cxcywh(pred_boxes, positive_boxes)
                max_iou = all_iou.max(dim=1)[0]  # [Q]
                no_overlap_mask = max_iou <= self.iou_thresh  # [Q] bool
            else:
                no_overlap_mask = torch.ones(num_preds, dtype=torch.bool, device=device)
            
            # 🔥 组合条件：高 VDVI + 不与 GT 重叠
            valid_mask = vdvi_mask & no_overlap_mask  # [Q] bool
            
            if valid_mask.sum() > 0:
                # 获取有效候选框及其分数
                valid_boxes = pred_boxes[valid_mask]  # [M, 4]
                valid_scores = pred_scores[valid_mask]  # [M]
                
                # 🔥 按分数排序，选取 top-k（向量化）
                num_to_select = min(num_samples, valid_boxes.shape[0])
                _, topk_indices = valid_scores.topk(num_to_select)
                neg_boxes = valid_boxes[topk_indices]  # [K, 4]
        
        # 🔥 如果预测框不足，从高 VDVI 区域随机采样补充（向量化）
        num_current = neg_boxes.shape[0]
        if num_current < num_samples:
            valid_positions = torch.nonzero(high_vdvi_mask, as_tuple=False)  # [P, 2]
            
            if valid_positions.shape[0] >= 10:
                num_to_sample = min(num_samples - num_current, valid_positions.shape[0] // 10, 100)
                if num_to_sample > 0:
                    # 🔥 向量化随机采样（避免 Python 循环）
                    indices = torch.randperm(valid_positions.shape[0], device=device)[:num_to_sample]
                    sampled_pos = valid_positions[indices]  # [S, 2] (y, x)
                    
                    # 🔥 向量化生成框（避免逐框创建）
                    cx_new = sampled_pos[:, 1].float() / W  # [S]
                    cy_new = sampled_pos[:, 0].float() / H  # [S]
                    
                    # 随机框大小（使用 torch 随机，避免 Python random）
                    w_new = torch.empty(num_to_sample, device=device).uniform_(self.min_neg_size, self.max_neg_size)
                    h_new = torch.empty(num_to_sample, device=device).uniform_(self.min_neg_size, self.max_neg_size)
                    
                    # 确保在图像内（向量化）
                    cx_new = cx_new.clamp(w_new / 2, 1 - w_new / 2)
                    cy_new = cy_new.clamp(h_new / 2, 1 - h_new / 2)
                    
                    # 堆叠为框 [S, 4]
                    random_boxes = torch.stack([cx_new, cy_new, w_new, h_new], dim=1)
                    
                    # 合并
                    neg_boxes = torch.cat([neg_boxes, random_boxes], dim=0) if neg_boxes.shape[0] > 0 else random_boxes
        
        # 🔥 检查是否有有效的负样本框
        if neg_boxes.shape[0] == 0:
            if return_boxes:
                return torch.zeros((0, encoder_feats[0].shape[1]), device=device), torch.zeros((0, 4), device=device)
            return torch.zeros((0, encoder_feats[0].shape[1]), device=device)
        
        # neg_boxes 已经是张量 [K, 4]，无需再 stack
        batch_ids = torch.full((neg_boxes.shape[0],), batch_id, device=device, dtype=torch.long)
        
        # 提取特征
        neg_feats = roi_extractor(
            encoder_feats=encoder_feats,
            boxes_cxcywh_norm=neg_boxes,
            batch_ids=batch_ids,
            img_size=img_size,
        )
        
        if return_boxes:
            return neg_feats, neg_boxes
        return neg_feats
    
    def mine_random_background(
        self,
        encoder_feats: List[torch.Tensor],
        positive_boxes: torch.Tensor,  # [N, 4] cxcywh normalized
        batch_id: int,
        img_size: Tuple[int, int],
        roi_extractor,  # RoI 特征提取函数
        return_boxes: bool = False,    # 🔥 是否返回框的位置（用于可视化）
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        从背景区域随机采样负样本
        
        Args:
            encoder_feats: 编码器特征列表
            positive_boxes: 正样本框（避免采样到这些区域）
            batch_id: batch 索引
            img_size: 图像尺寸 (H, W)
            roi_extractor: RoI 特征提取函数
        
        Returns:
            neg_feats: 负样本特征 [M, D]
        """
        device = encoder_feats[0].device
        H, W = img_size
        
        neg_boxes = []
        attempts = 0
        max_attempts = self.num_neg_samples * 10
        
        while len(neg_boxes) < self.num_neg_samples and attempts < max_attempts:
            attempts += 1
            
            # 随机生成框
            w = random.uniform(self.min_neg_size, self.max_neg_size)
            h = random.uniform(self.min_neg_size, self.max_neg_size)
            cx = random.uniform(w/2, 1 - w/2)
            cy = random.uniform(h/2, 1 - h/2)
            
            neg_box = torch.tensor([cx, cy, w, h], device=device)
            
            # 检查是否与正样本重叠
            if positive_boxes.shape[0] > 0:
                iou = self._box_iou_cxcywh(neg_box.unsqueeze(0), positive_boxes)
                if iou.max() > self.iou_thresh:
                    continue  # 与正样本重叠，跳过
            
            neg_boxes.append(neg_box)
        
        if len(neg_boxes) == 0:
            if return_boxes:
                return torch.zeros((0, encoder_feats[0].shape[1]), device=device), torch.zeros((0, 4), device=device)
            return torch.zeros((0, encoder_feats[0].shape[1]), device=device)
        
        neg_boxes = torch.stack(neg_boxes, dim=0)  # [M, 4]
        batch_ids = torch.full((neg_boxes.shape[0],), batch_id, device=device, dtype=torch.long)
        
        # 提取特征
        neg_feats = roi_extractor(
            encoder_feats=encoder_feats,
            boxes_cxcywh_norm=neg_boxes,
            batch_ids=batch_ids,
            img_size=img_size,
        )
        
        if return_boxes:
            return neg_feats, neg_boxes
        return neg_feats
    
    def mine_low_confidence(
        self,
        encoder_feats: List[torch.Tensor],
        all_boxes: torch.Tensor,      # [Q, 4] 所有预测框
        all_scores: torch.Tensor,     # [Q] 所有分数
        conf_thresh: float,           # 置信度阈值
        batch_id: int,
        img_size: Tuple[int, int],
        roi_extractor,
    ) -> torch.Tensor:
        """
        从低置信度预测中挖掘负样本
        
        思想：置信度在 [low_thresh, conf_thresh) 之间的预测
        - 模型认为"可能是目标"但不够确定
        - 这些是"硬负样本"，对学习最有价值
        
        Args:
            all_boxes: 所有预测框 [Q, 4]
            all_scores: 所有预测分数 [Q]
            conf_thresh: 伪标签置信度阈值
        
        Returns:
            neg_feats: 负样本特征 [M, D]
        """
        device = encoder_feats[0].device
        
        # 选择置信度在 [0.1, conf_thresh) 之间的框作为硬负样本
        low_thresh = 0.1
        hard_neg_mask = (all_scores >= low_thresh) & (all_scores < conf_thresh)
        
        if hard_neg_mask.sum() == 0:
            return torch.zeros((0, encoder_feats[0].shape[1]), device=device)
        
        hard_neg_boxes = all_boxes[hard_neg_mask]
        
        # 限制数量
        if hard_neg_boxes.shape[0] > self.num_neg_samples:
            # 选择分数最高的（最接近阈值的，最难的负样本）
            hard_neg_scores = all_scores[hard_neg_mask]
            _, topk_idx = hard_neg_scores.topk(self.num_neg_samples)
            hard_neg_boxes = hard_neg_boxes[topk_idx]
        
        batch_ids = torch.full((hard_neg_boxes.shape[0],), batch_id, device=device, dtype=torch.long)
        
        # 提取特征
        neg_feats = roi_extractor(
            encoder_feats=encoder_feats,
            boxes_cxcywh_norm=hard_neg_boxes,
            batch_ids=batch_ids,
            img_size=img_size,
        )
        
        return neg_feats
    
    def _box_iou_cxcywh(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """计算 cxcywh 格式框的 IoU"""
        # 转换为 xyxy
        def cxcywh_to_xyxy(boxes):
            cx, cy, w, h = boxes.unbind(-1)
            return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
        
        boxes1_xyxy = cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = cxcywh_to_xyxy(boxes2)
        
        # 计算交集
        lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
        rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        # 计算并集
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        
        return inter / union.clamp(min=1e-6)


class WeakStrongContrastiveLoss(nn.Module):
    """
    Weak-Strong 一致性对比损失
    
    思想：同一目标在 weak 和 strong 增强下的特征应该一致
    
    这是 Semi-DETR 中已有的一致性损失的对比学习版本，
    可以作为补充损失使用
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.contrastive_loss = ContrastiveLoss(temperature)
    
    def forward(
        self,
        weak_feats: torch.Tensor,    # [N, D] weak 增强下的目标特征
        strong_feats: torch.Tensor,  # [N, D] strong 增强下的同一目标特征
        negative_feats: Optional[torch.Tensor] = None,  # [M, D] 负样本特征
    ) -> torch.Tensor:
        """
        计算 weak-strong 一致性对比损失
        
        Args:
            weak_feats: weak 增强下的特征（作为 anchor）
            strong_feats: strong 增强下的特征（作为 positive）
            negative_feats: 可选的负样本特征
        
        Returns:
            loss: 标量损失
        """
        if weak_feats.shape[0] == 0:
            return torch.tensor(0.0, device=weak_feats.device, requires_grad=True)
        
        if negative_feats is None or negative_feats.shape[0] == 0:
            # 如果没有负样本，使用 batch 内其他样本作为负样本
            # 但需要排除对应的正样本对
            # 简化版本：只计算正样本对的相似度损失
            weak_feats = F.normalize(weak_feats, dim=-1)
            strong_feats = F.normalize(strong_feats, dim=-1)
            pos_sim = (weak_feats * strong_feats).sum(dim=-1)
            loss = (1 - pos_sim).mean()
            return loss
        
        return self.contrastive_loss(weak_feats, strong_feats, negative_feats)


def visualize_negative_samples(
    rgb_image: torch.Tensor,          # [3, H, W] 归一化的 RGB 图像
    gt_boxes: torch.Tensor,           # [N, 4] GT 框（正样本）cxcywh normalized
    vdvi_neg_boxes: torch.Tensor,     # [M1, 4] VDVI 硬负样本框 cxcywh normalized
    bg_neg_boxes: torch.Tensor,       # [M2, 4] 背景负样本框（已废弃，传入空张量）
    save_path: str,                    # 保存路径
    img_size: Tuple[int, int],        # 图像尺寸 (H, W)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> None:
    """
    可视化对比学习的正负样本
    
    Args:
        rgb_image: 归一化的 RGB 图像 [3, H, W]
        gt_boxes: GT 框（正样本）[N, 4] (cxcywh normalized) - 目标杂草
        vdvi_neg_boxes: VDVI 硬负样本框 [M1, 4] (cxcywh normalized) - 玉米叶子等
        bg_neg_boxes: 背景负样本框（已废弃，不再使用）
        save_path: 保存路径
        img_size: 图像尺寸 (H, W)
    """
    H, W = img_size
    
    # 🔥 预加载字体（只加载一次，避免重复 I/O）
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_small = ImageFont.load_default()
        font = ImageFont.load_default()
    
    # 反归一化图像（确保所有张量在同一设备上）
    device = rgb_image.device
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t = torch.tensor(std, device=device).view(3, 1, 1)
    img_denorm = rgb_image * std_t + mean_t
    img_denorm = torch.clamp(img_denorm, 0, 1)
    
    # 转换为 PIL Image
    img_np = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    
    # 🔥 批量转换框坐标到 CPU（避免逐框转换）
    def boxes_to_pixel_coords(boxes, H, W):
        """将 cxcywh normalized 格式的框批量转换为像素坐标 [x1, y1, x2, y2]"""
        if boxes.shape[0] == 0:
            return []
        boxes_cpu = boxes.detach().cpu()
        cx, cy, w, h = boxes_cpu[:, 0], boxes_cpu[:, 1], boxes_cpu[:, 2], boxes_cpu[:, 3]
        x1 = ((cx - w/2) * W).int().clamp(0, W)
        y1 = ((cy - h/2) * H).int().clamp(0, H)
        x2 = ((cx + w/2) * W).int().clamp(0, W)
        y2 = ((cy + h/2) * H).int().clamp(0, H)
        return list(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist()))
    
    # 🔥 绘制正样本框（黄色 - 目标杂草）
    if isinstance(gt_boxes, torch.Tensor) and gt_boxes.numel() > 0:
        if gt_boxes.dim() == 1:
            gt_boxes = gt_boxes.unsqueeze(0)
        pixel_coords = boxes_to_pixel_coords(gt_boxes, H, W)
        for x1, y1, x2, y2 in pixel_coords:
            if x2 > x1 and y2 > y1:
                draw.rectangle([x1, y1, x2, y2], outline='yellow', width=3)
                draw.text((x1 + 2, y1 + 2), "Pos", fill='yellow', font=font_small)
    
    # 🔥 绘制负样本框（红色 - 玉米叶子等硬负样本）
    if isinstance(vdvi_neg_boxes, torch.Tensor) and vdvi_neg_boxes.numel() > 0:
        if vdvi_neg_boxes.dim() == 1:
            vdvi_neg_boxes = vdvi_neg_boxes.unsqueeze(0)
        pixel_coords = boxes_to_pixel_coords(vdvi_neg_boxes, H, W)
        for x1, y1, x2, y2 in pixel_coords:
            if x2 > x1 and y2 > y1:
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                draw.text((x1 + 2, y1 + 2), "Neg", fill='red', font=font_small)
    
    # 图例位置（显示在图像右上角）
    legend_x = W - 280
    legend_y = 10
    # 绘制图例背景（半透明）
    draw.rectangle([legend_x - 5, legend_y - 5, W - 5, legend_y + 50], fill=(0, 0, 0, 180), outline='white', width=1)
    draw.text((legend_x, legend_y), "Positive (Yellow): Target Weeds", fill='yellow', font=font)
    draw.text((legend_x, legend_y + 20), "Negative (Red): Corn Leaves", fill='red', font=font)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_pil.save(save_path)


def compute_contrastive_loss_for_ssod(
    proto_state,                      # ProtoState 实例
    encoder_feats: List[torch.Tensor], # 编码器特征
    gt_boxes: torch.Tensor,           # GT 框 [N, 4] cxcywh normalized
    gt_batch_ids: torch.Tensor,       # GT 框的 batch 索引 [N]
    all_pred_boxes: torch.Tensor,     # 所有预测框 [B, Q, 4]
    all_pred_scores: torch.Tensor,    # 所有预测分数 [B, Q]
    conf_thresh: float,               # 伪标签置信度阈值
    img_size: Tuple[int, int],        # 图像尺寸 (H, W)
    roi_extractor,                    # RoI 特征提取函数
    contrastive_weight: float = 0.1,  # 对比损失权重
    temperature: float = 0.1,
    num_neg_samples: int = 16,
    vdvi_maps: Optional[torch.Tensor] = None,  # 🔥 新增：VDVI 图 [B, 1, H, W]
    vdvi_thresh: float = 0.03,                  # 🔥 新增：VDVI 阈值
    rgb_images: Optional[torch.Tensor] = None,  # 🔥 新增：RGB 图像 [B, 3, H, W] 用于可视化
    visualize: bool = False,                     # 🔥 是否可视化
    vis_save_dir: Optional[str] = None,          # 🔥 可视化保存目录
    epoch: int = 0,                              # 🔥 当前 epoch
    iteration: int = 0,                          # 🔥 当前 iteration
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算半监督训练中的对比损失
    
    🔥 改进的负样本挖掘策略：
    1. 正样本：GT 框（目标杂草）→ 拉近原型
    2. 硬负样本：高 VDVI 但不是目标（玉米叶子）→ 推远原型  ← 最重要！
    
    注意：不再使用背景负样本（土壤等），因为：
    - 背景与目标差异太明显，属于简单负样本，对训练帮助不大
    - 真正的挑战是区分"目标杂草 vs 玉米叶子"，需要硬负样本
    - 专注于硬负样本可以让模型更聚焦于困难区分任务
    
    Args:
        proto_state: 语义原型状态
        encoder_feats: 编码器特征列表
        gt_boxes: 真实框 [N, 4]
        gt_batch_ids: 真实框的 batch 索引 [N]
        all_pred_boxes: 所有预测框 [B, Q, 4]
        all_pred_scores: 所有预测分数 [B, Q]
        conf_thresh: 置信度阈值
        img_size: 图像尺寸
        roi_extractor: RoI 特征提取函数
        contrastive_weight: 对比损失权重
        temperature: 温度参数
        num_neg_samples: 每个 batch 采样的负样本数量
        vdvi_maps: 🔥 VDVI 图（用于挖掘玉米叶子等硬负样本）
        vdvi_thresh: 🔥 VDVI 阈值
    
    Returns:
        loss: 对比损失（已乘以权重）
        stats: 统计信息
    """
    device = encoder_feats[0].device
    stats = {}
    
    # 检查原型是否已初始化
    if proto_state is None or not proto_state.is_initialized():
        return torch.tensor(0.0, device=device, requires_grad=True), {'contrastive_loss': 0.0}
    
    prototype = proto_state.proto  # [D]
    
    # ========== 1. 提取正样本特征（GT 框 = 目标杂草）==========
    if gt_boxes.shape[0] > 0:
        positive_feats = roi_extractor(
            encoder_feats=encoder_feats,
            boxes_cxcywh_norm=gt_boxes,
            batch_ids=gt_batch_ids,
            img_size=img_size,
        )
    else:
        positive_feats = torch.zeros((0, prototype.shape[0]), device=device)
    
    # ========== 2. 挖掘负样本 ==========
    neg_miner = NegativeSampleMiner(num_neg_samples=num_neg_samples)
    
    B = all_pred_boxes.shape[0]
    all_negative_feats = []
    num_vdvi_neg = 0
    
    # 🔥 收集正负样本框用于可视化
    all_vdvi_neg_boxes = []  # 每个 batch 的 VDVI 硬负样本框
    all_positive_boxes = []   # 每个 batch 的正样本框（GT框）
    
    for b in range(B):
        # 获取当前 batch 的 GT 框
        batch_mask = gt_batch_ids == b
        batch_gt_boxes = gt_boxes[batch_mask] if batch_mask.sum() > 0 else torch.zeros((0, 4), device=device)
        
        # 🔥 收集正样本框用于可视化（每个 batch 都要收集，即使没有 GT 框）
        if visualize:
            all_positive_boxes.append(batch_gt_boxes)
        
        # 🔥 只使用 VDVI 硬负样本（玉米叶子 = 高 VDVI 但不是目标）
        # 移除背景负样本：背景（土壤）与目标差异太明显，属于简单负样本，对训练帮助不大
        # 真正的挑战是区分"目标杂草 vs 玉米叶子"，需要硬负样本
        vdvi_neg_boxes_b = torch.zeros((0, 4), device=device)
        if vdvi_maps is not None and vdvi_maps.shape[0] > b:
            vdvi_map = vdvi_maps[b, 0]  # [H, W]
            
            # 🔥 获取当前 batch 的预测框和分数（用于置信度筛选）
            batch_pred_boxes = all_pred_boxes[b] if all_pred_boxes.shape[0] > b else None
            batch_pred_scores = all_pred_scores[b] if all_pred_scores.shape[0] > b else None
            
            result = neg_miner.mine_vdvi_hard_negatives(
                encoder_feats=encoder_feats,
                vdvi_map=vdvi_map,
                positive_boxes=batch_gt_boxes,
                batch_id=b,
                img_size=img_size,
                roi_extractor=roi_extractor,
                vdvi_thresh=vdvi_thresh,
                num_samples=num_neg_samples // B,  # 🔥 每张图的负样本数 = 总数 / batch_size
                return_boxes=visualize,  # 🔥 需要可视化时返回框
                pred_boxes=batch_pred_boxes,     # 🔥 传入预测框
                pred_scores=batch_pred_scores,   # 🔥 传入预测分数
            )
            if visualize:
                vdvi_hard_feats, vdvi_neg_boxes_b = result
            else:
                vdvi_hard_feats = result
            
            if vdvi_hard_feats.shape[0] > 0:
                all_negative_feats.append(vdvi_hard_feats)
                num_vdvi_neg += vdvi_hard_feats.shape[0]
                if visualize:
                    all_vdvi_neg_boxes.append(vdvi_neg_boxes_b)
        
        # 🔥 确保每个 batch 都有对应的负样本框列表（用于可视化）
        if visualize and (vdvi_maps is None or vdvi_maps.shape[0] <= b or vdvi_hard_feats.shape[0] == 0):
            # 如果没有 VDVI 负样本，添加空列表用于可视化
            if len(all_vdvi_neg_boxes) <= b:
                all_vdvi_neg_boxes.append(torch.zeros((0, 4), device=device))
    
    if len(all_negative_feats) > 0:
        negative_feats = torch.cat(all_negative_feats, dim=0)
    else:
        negative_feats = torch.zeros((0, prototype.shape[0]), device=device)
    
    # 🔥 记录负样本来源统计
    stats['num_vdvi_neg'] = num_vdvi_neg
    
    # 🔥 可视化正负样本（仅在需要时）
    if visualize and rgb_images is not None and vis_save_dir is not None:
        for b in range(min(B, rgb_images.shape[0])):
            # 获取该 batch 的正样本框（GT框）
            positive_boxes_b = all_positive_boxes[b] if b < len(all_positive_boxes) else torch.zeros((0, 4), device=device)
            
            # 获取该 batch 的 VDVI 硬负样本框
            vdvi_boxes_b = all_vdvi_neg_boxes[b] if b < len(all_vdvi_neg_boxes) else torch.zeros((0, 4), device=device)
            
            # 🔥 调试信息：打印正负样本数量
            num_pos = positive_boxes_b.shape[0] if isinstance(positive_boxes_b, torch.Tensor) else 0
            num_neg = vdvi_boxes_b.shape[0] if isinstance(vdvi_boxes_b, torch.Tensor) else 0
            if num_pos > 0 or num_neg > 0:
                print(f"[可视化] Epoch {epoch}, Iter {iteration}, Batch {b}: 正样本={num_pos}, 负样本={num_neg}")
            
            # 保存路径（更新文件名，反映正负样本可视化）
            save_path = os.path.join(vis_save_dir, f"contrastive_samples_epoch{epoch}_iter{iteration}_batch{b}.png")
            
            # 可视化正负样本
            visualize_negative_samples(
                rgb_image=rgb_images[b],
                gt_boxes=positive_boxes_b,  # 🔥 正样本框（GT框）
                vdvi_neg_boxes=vdvi_boxes_b,  # 🔥 负样本框（VDVI硬负样本）
                bg_neg_boxes=torch.zeros((0, 4), device=device),  # 🔥 不再使用背景负样本
                save_path=save_path,
                img_size=img_size,
            )
    
    # ========== 3. 计算对比损失 ==========
    proto_contrast = PrototypeContrastiveLoss(temperature=temperature)
    loss, contrast_stats = proto_contrast(
        prototype=prototype,
        positive_feats=positive_feats,
        negative_feats=negative_feats,
    )
    
    # 加权
    weighted_loss = loss * contrastive_weight
    
    # 更新统计信息
    stats['contrastive_loss'] = weighted_loss.item()
    stats['num_positive'] = positive_feats.shape[0]
    stats['num_negative'] = negative_feats.shape[0]
    stats.update(contrast_stats)
    
    return weighted_loss, stats

