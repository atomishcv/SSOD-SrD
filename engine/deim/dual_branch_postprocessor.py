"""
Dual-Branch PostProcessor for Multi-Granularity Object Detection
双分支后处理器：处理单株和群落分支的输出，合并预测结果

核心功能：
1. 分支专用NMS：单株和群落分支分别进行NMS
2. 类型感知合并：根据类型标记合并两个分支的输出
3. 重叠抑制：群落框内不输出过多单株框
4. 置信度校准：根据分支类型调整置信度

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Dict, List, Tuple, Optional

from ..core import register

__all__ = ['DualBranchPostProcessor']


@register()
class DualBranchPostProcessor(nn.Module):
    """
    双分支后处理器
    """
    
    __share__ = ['num_classes', 'remap_mscoco_category']
    
    def __init__(self,
                 num_classes: int = 80,
                 use_focal_loss: bool = True,
                 num_top_queries: int = 300,
                 remap_mscoco_category: bool = False,
                 # 分支专用配置
                 nms_threshold_individual: float = 0.5,
                 nms_threshold_colony: float = 0.3,
                 score_threshold_individual: float = 0.3,
                 score_threshold_colony: float = 0.2,
                 # 重叠抑制
                 suppress_overlap: bool = True,
                 overlap_iou_threshold: float = 0.7,
                 # 输出类型
                 output_type_label: bool = True):
        """
        Args:
            nms_threshold_individual: 单株分支NMS阈值
            nms_threshold_colony: 群落分支NMS阈值（更低以保留更多大框）
            score_threshold_individual: 单株分支置信度阈值
            score_threshold_colony: 群落分支置信度阈值（更低以保留更多群落）
            suppress_overlap: 是否抑制群落框内的单株框
            overlap_iou_threshold: 重叠抑制的IoU阈值
            output_type_label: 是否输出类型标签
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.remap_mscoco_category = remap_mscoco_category
        
        self.nms_threshold_individual = nms_threshold_individual
        self.nms_threshold_colony = nms_threshold_colony
        self.score_threshold_individual = score_threshold_individual
        self.score_threshold_colony = score_threshold_colony
        
        self.suppress_overlap = suppress_overlap
        self.overlap_iou_threshold = overlap_iou_threshold
        self.output_type_label = output_type_label
        
        # MSCOCO类别映射
        self.deploy_mode = False
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                orig_target_sizes: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        后处理双分支输出
        
        Args:
            outputs: 模型输出，包含两个分支的预测
            orig_target_sizes: 原始图像尺寸 [B, 2]
        
        Returns:
            results: 每张图像的检测结果列表
        """
        # 获取分支输出
        if 'pred_logits_individual' in outputs:
            logits_ind = outputs['pred_logits_individual']
            boxes_ind = outputs['pred_boxes_individual']
            logits_col = outputs['pred_logits_colony']
            boxes_col = outputs['pred_boxes_colony']
        else:
            # 从合并输出分离（使用 pred_types 来确定分割点）
            logits = outputs['pred_logits']
            boxes = outputs['pred_boxes']
            
            if 'pred_types' in outputs:
                # 根据类型标记分割：type=0 是单株，type=1 是群落
                types = outputs['pred_types']
                num_q_ind = (types[0] == 0).sum().item()
            else:
                # 回退到默认的 2:1 分割（200 单株 + 100 群落）
                num_q_ind = logits.shape[1] * 2 // 3
            
            logits_ind = logits[:, :num_q_ind]
            boxes_ind = boxes[:, :num_q_ind]
            logits_col = logits[:, num_q_ind:]
            boxes_col = boxes[:, num_q_ind:]
        
        # 获取类型标记
        if 'pred_types' in outputs:
            types = outputs['pred_types']
        else:
            batch_size = logits_ind.shape[0]
            num_q_ind = logits_ind.shape[1]
            num_q_col = logits_col.shape[1]
            types = torch.cat([
                torch.zeros(batch_size, num_q_ind, device=logits_ind.device),
                torch.ones(batch_size, num_q_col, device=logits_col.device)
            ], dim=1)
        
        batch_size = logits_ind.shape[0]
        results = []
        
        for b in range(batch_size):
            # 处理单株分支
            ind_result = self._process_branch(
                logits_ind[b], boxes_ind[b],
                orig_target_sizes[b],
                self.score_threshold_individual,
                self.nms_threshold_individual,
                target_type=0
            )
            
            # 处理群落分支
            col_result = self._process_branch(
                logits_col[b], boxes_col[b],
                orig_target_sizes[b],
                self.score_threshold_colony,
                self.nms_threshold_colony,
                target_type=1
            )
            
            # 合并结果
            merged_result = self._merge_results(ind_result, col_result, orig_target_sizes[b])
            results.append(merged_result)
        
        return results
    
    def _process_branch(self,
                       logits: torch.Tensor,
                       boxes: torch.Tensor,
                       orig_size: torch.Tensor,
                       score_threshold: float,
                       nms_threshold: float,
                       target_type: int) -> Dict[str, torch.Tensor]:
        """处理单个分支的输出"""
        # 计算分数
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=-1)[:, :-1]
        
        # 获取最大分数和对应类别
        max_scores, labels = scores.max(dim=-1)
        
        # 过滤低置信度预测
        keep = max_scores > score_threshold
        
        if keep.sum() == 0:
            return {
                'scores': torch.tensor([], device=logits.device),
                'labels': torch.tensor([], dtype=torch.long, device=logits.device),
                'boxes': torch.tensor([], device=logits.device).reshape(0, 4),
                'types': torch.tensor([], dtype=torch.long, device=logits.device)
            }
        
        scores_kept = max_scores[keep]
        labels_kept = labels[keep]
        boxes_kept = boxes[keep]
        
        # 转换框坐标到原始图像尺寸
        img_h, img_w = orig_size.unbind()
        scale = torch.stack([img_w, img_h, img_w, img_h], dim=0)
        boxes_scaled = boxes_kept * scale
        
        # 转换为xyxy格式
        boxes_xyxy = self._box_cxcywh_to_xyxy(boxes_scaled)
        
        # NMS
        keep_nms = torchvision.ops.nms(boxes_xyxy, scores_kept, nms_threshold)
        
        return {
            'scores': scores_kept[keep_nms],
            'labels': labels_kept[keep_nms],
            'boxes': boxes_xyxy[keep_nms],
            'types': torch.full_like(labels_kept[keep_nms], target_type)
        }
    
    def _merge_results(self,
                      ind_result: Dict[str, torch.Tensor],
                      col_result: Dict[str, torch.Tensor],
                      orig_size: torch.Tensor) -> Dict[str, torch.Tensor]:
        """合并两个分支的结果"""
        # 简单拼接
        if len(ind_result['scores']) == 0 and len(col_result['scores']) == 0:
            device = orig_size.device
            return {
                'scores': torch.tensor([], device=device),
                'labels': torch.tensor([], dtype=torch.long, device=device),
                'boxes': torch.tensor([], device=device).reshape(0, 4),
                'types': torch.tensor([], dtype=torch.long, device=device) if self.output_type_label else None
            }
        
        if len(ind_result['scores']) == 0:
            result = col_result
        elif len(col_result['scores']) == 0:
            result = ind_result
        else:
            # 重叠抑制：群落框内的单株框降低权重或移除
            if self.suppress_overlap:
                ind_result = self._suppress_overlapping(ind_result, col_result)
            
            # 合并
            result = {
                'scores': torch.cat([ind_result['scores'], col_result['scores']]),
                'labels': torch.cat([ind_result['labels'], col_result['labels']]),
                'boxes': torch.cat([ind_result['boxes'], col_result['boxes']]),
                'types': torch.cat([ind_result['types'], col_result['types']])
            }
        
        # 按分数排序并限制数量
        if len(result['scores']) > self.num_top_queries:
            topk_indices = result['scores'].topk(self.num_top_queries).indices
            result = {k: v[topk_indices] if v is not None else None for k, v in result.items()}
        
        if not self.output_type_label:
            result.pop('types', None)
        
        return result
    
    def _suppress_overlapping(self,
                             ind_result: Dict[str, torch.Tensor],
                             col_result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """抑制被群落框覆盖的单株框"""
        if len(ind_result['scores']) == 0 or len(col_result['scores']) == 0:
            return ind_result
        
        ind_boxes = ind_result['boxes']
        col_boxes = col_result['boxes']
        
        # 计算IoU
        iou_matrix = self._box_iou(ind_boxes, col_boxes)
        
        # 找到被群落框高度覆盖的单株框
        max_iou, _ = iou_matrix.max(dim=1)
        keep = max_iou < self.overlap_iou_threshold
        
        return {
            'scores': ind_result['scores'][keep],
            'labels': ind_result['labels'][keep],
            'boxes': ind_result['boxes'][keep],
            'types': ind_result['types'][keep]
        }
    
    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """将cxcywh格式转换为xyxy格式"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    @staticmethod
    def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """计算IoU矩阵"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / union.clamp(min=1e-6)
        
        return iou
    
    def deploy(self):
        """切换到部署模式"""
        self.deploy_mode = True
        return self

