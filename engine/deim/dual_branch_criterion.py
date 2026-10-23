"""
Triple-Branch DEIM Criterion for Multi-Granularity Object Detection
基于目标框尺寸的三分支损失计算

核心设计：
1. 单株分支 (individual): 只计算小尺寸目标的损失
2. 集群分支 (colony): 只计算大尺寸目标的损失  
3. 综合分支 (combined): 计算所有目标的损失

尺寸划分基于数据集目标框的大小分布（面积比例）

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy
from typing import List, Dict, Tuple, Optional

from .dfine_utils import bbox2distance
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register


@register()
class TripleBranchCriterion(nn.Module):
    """
    三分支损失计算器
    
    根据目标框尺寸将GT分配到不同分支：
    - individual: 面积 < size_threshold 的小目标
    - colony: 面积 >= size_threshold 的大目标
    - combined: 所有目标
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self,
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 num_classes=80,
                 reg_max=32,
                 boxes_weight_format=None,
                 share_matched_indices=False,
                 mal_alpha=None,
                 use_uni_set=True,
                 # 尺寸划分阈值（基于归一化面积 w*h）
                 size_threshold: float = 0.01,  # 默认1%的图像面积
                 # 分支损失权重
                 branch_weights: Dict[str, float] = None,
                 ):
        """
        Args:
            size_threshold: 小/大目标的划分阈值（归一化面积）
                           面积 < threshold 为小目标 -> individual
                           面积 >= threshold 为大目标 -> colony
            branch_weights: 各分支损失的权重，默认 {'individual': 1.0, 'colony': 1.0, 'combined': 1.0}
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        
        # 尺寸划分阈值
        self.size_threshold = size_threshold
        
        # 分支权重
        self.branch_weights = branch_weights or {
            'individual': 1.0,
            'colony': 1.0,
            'combined': 1.0
        }
        
        # 缓存
        self.fgl_targets = {}
        self.own_targets = {}
        self.num_pos, self.num_neg = None, None
    
    def _get_all_loss_keys(self, branch_key: str, num_aux: int, has_pre: bool, 
                           num_dn: int = 0, has_dn_pre: bool = False) -> List[str]:
        """
        获取某个分支的所有可能的 loss keys
        🔥 用于确保所有 GPU 返回相同的 keys 集合
        """
        keys = []
        
        # 主损失 keys
        for loss_name in self.losses:
            if loss_name == 'boxes':
                keys.append(f'loss_bbox_{branch_key}')
                keys.append(f'loss_giou_{branch_key}')
            elif loss_name == 'local':
                keys.append(f'loss_fgl_{branch_key}')
            else:
                keys.append(f'loss_{loss_name}_{branch_key}')
        
        # 辅助层损失 keys
        for i in range(num_aux):
            for loss_name in self.losses:
                if loss_name == 'boxes':
                    keys.append(f'loss_bbox_{branch_key}_aux_{i}')
                    keys.append(f'loss_giou_{branch_key}_aux_{i}')
                elif loss_name == 'local':
                    keys.append(f'loss_fgl_{branch_key}_aux_{i}')
                else:
                    keys.append(f'loss_{loss_name}_{branch_key}_aux_{i}')
        
        # Pre 输出损失 keys
        if has_pre:
            for loss_name in self.losses:
                if loss_name == 'boxes':
                    keys.append(f'loss_bbox_{branch_key}_pre')
                    keys.append(f'loss_giou_{branch_key}_pre')
                elif loss_name == 'local':
                    keys.append(f'loss_fgl_{branch_key}_pre')
                else:
                    keys.append(f'loss_{loss_name}_{branch_key}_pre')
        
        # 去噪损失 keys (仅 combined 分支)
        if branch_key == 'combined' and num_dn > 0:
            for i in range(num_dn):
                for loss_name in self.losses:
                    if loss_name == 'boxes':
                        keys.append(f'loss_bbox_{branch_key}_dn_{i}')
                        keys.append(f'loss_giou_{branch_key}_dn_{i}')
                    elif loss_name == 'local':
                        keys.append(f'loss_fgl_{branch_key}_dn_{i}')
                    else:
                        keys.append(f'loss_{loss_name}_{branch_key}_dn_{i}')
            
            if has_dn_pre:
                for loss_name in self.losses:
                    if loss_name == 'boxes':
                        keys.append(f'loss_bbox_{branch_key}_dn_pre')
                        keys.append(f'loss_giou_{branch_key}_dn_pre')
                    elif loss_name == 'local':
                        keys.append(f'loss_fgl_{branch_key}_dn_pre')
                    else:
                        keys.append(f'loss_{loss_name}_{branch_key}_dn_pre')
        
        return keys

    def _split_targets_by_size(self, targets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        根据目标框尺寸将targets分割为小目标和大目标
        优先使用 size_category 字段（如果可用），否则使用面积计算
        
        Args:
            targets: 原始targets列表
            
        Returns:
            targets_small: 小目标（用于individual分支）
            targets_large: 大目标（用于colony分支）
        """
        targets_small = []
        targets_large = []
        
        for t in targets:
            boxes = t['boxes']  # [N, 4] (cx, cy, w, h) 归一化坐标
            labels = t['labels']  # [N]
            
            if len(boxes) == 0:
                # 空目标
                targets_small.append({'boxes': boxes, 'labels': labels})
                targets_large.append({'boxes': boxes, 'labels': labels})
                continue
            
            # 优先使用 size_category 字段（如果可用）
            if 'size_categories' in t:
                size_categories = t['size_categories']  # [N] tensor: 0=small, 1=big, -1=unknown
                small_mask = size_categories == 0
                large_mask = size_categories == 1
                # 对于 unknown (-1) 的目标，使用面积计算作为回退
                unknown_mask = size_categories == -1
                if unknown_mask.any():
                    areas = boxes[:, 2] * boxes[:, 3]  # w * h
                    unknown_small = unknown_mask & (areas < self.size_threshold)
                    unknown_large = unknown_mask & (areas >= self.size_threshold)
                    small_mask = small_mask | unknown_small
                    large_mask = large_mask | unknown_large
            else:
                # 回退到面积计算
                areas = boxes[:, 2] * boxes[:, 3]  # w * h
                small_mask = areas < self.size_threshold
                large_mask = areas >= self.size_threshold
            
            targets_small.append({
                'boxes': boxes[small_mask],
                'labels': labels[small_mask]
            })
            targets_large.append({
                'boxes': boxes[large_mask],
                'labels': labels[large_mask]
            })
        
        return targets_small, targets_large

    def _get_branch_outputs(self, outputs: Dict, branch: str) -> Dict:
        """
        获取指定分支的输出
        
        Args:
            outputs: 模型完整输出
            branch: 'individual', 'colony', 或 'combined'
        """
        suffix = f'_{branch}'
        branch_outputs = {}
        
        # 主输出
        if f'pred_logits{suffix}' in outputs:
            branch_outputs['pred_logits'] = outputs[f'pred_logits{suffix}']
            branch_outputs['pred_boxes'] = outputs[f'pred_boxes{suffix}']
        
        if f'pred_corners{suffix}' in outputs:
            branch_outputs['pred_corners'] = outputs[f'pred_corners{suffix}']
            branch_outputs['ref_points'] = outputs[f'ref_points{suffix}']
        
        # 复制共享参数
        if 'up' in outputs:
            branch_outputs['up'] = outputs['up']
        if 'reg_scale' in outputs:
            branch_outputs['reg_scale'] = outputs['reg_scale']
        
        return branch_outputs

    def _get_branch_aux_outputs(self, outputs: Dict, branch: str) -> List[Dict]:
        """获取指定分支的辅助输出"""
        key = f'aux_outputs_{branch}'
        if key in outputs:
            aux_list = outputs[key]
            # 添加共享参数
            for aux in aux_list:
                if 'up' in outputs:
                    aux['up'] = outputs['up']
                if 'reg_scale' in outputs:
                    aux['reg_scale'] = outputs['reg_scale']
            return aux_list
        return []

    def _get_branch_pre_outputs(self, outputs: Dict, branch: str) -> Optional[Dict]:
        """获取指定分支的pre输出"""
        key = f'pre_outputs_{branch}'
        return outputs.get(key, None)

    def _get_branch_dn_outputs(self, outputs: Dict, branch: str) -> Tuple[Optional[List], Optional[Dict], Optional[Dict]]:
        """获取指定分支的去噪输出"""
        dn_key = f'dn_outputs_{branch}'
        dn_pre_key = f'dn_pre_outputs_{branch}'
        dn_meta_key = f'dn_meta_{branch}'
        
        dn_outputs = outputs.get(dn_key, None)
        dn_pre_outputs = outputs.get(dn_pre_key, None)
        dn_meta = outputs.get(dn_meta_key, None)
        
        if dn_outputs is not None and 'up' in outputs:
            for aux in dn_outputs:
                aux['up'] = outputs['up']
                aux['reg_scale'] = outputs['reg_scale']
                aux['is_dn'] = True
        
        return dn_outputs, dn_pre_outputs, dn_meta

    # ========== 损失函数 ==========
    
    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            if len(target_boxes) == 0:
                return {'loss_vfl': torch.tensor(0.0, device=outputs['pred_logits'].device)}
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            if len(target_boxes) == 0:
                return {'loss_mal': torch.tensor(0.0, device=outputs['pred_logits'].device)}
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha is not None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        if len(target_boxes) == 0:
            device = outputs['pred_boxes'].device
            return {'loss_bbox': torch.tensor(0.0, device=device), 
                    'loss_giou': torch.tensor(0.0, device=device)}
        
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, branch_key='combined', T=5):
        losses = {}
        if 'pred_corners' not in outputs:
            return losses
            
        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        if len(target_boxes) == 0:
            return losses

        pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
        ref_points = outputs['ref_points'][idx].detach()
        
        cache_key = f'{branch_key}_dn' if 'is_dn' in outputs else branch_key
        
        with torch.no_grad():
            if cache_key not in self.fgl_targets:
                self.fgl_targets[cache_key] = bbox2distance(
                    ref_points, box_cxcywh_to_xyxy(target_boxes),
                    self.reg_max, outputs['reg_scale'], outputs['up'])

        target_corners, weight_right, weight_left = self.fgl_targets[cache_key]

        ious = torch.diag(box_iou(
            box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), 
            box_cxcywh_to_xyxy(target_boxes))[0])
        weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

        losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
            pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

        if 'teacher_corners' in outputs:
            pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
            target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
            if not torch.equal(pred_corners, target_corners):
                weight_targets_local = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]

                mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                mask[idx] = True
                mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(weight_targets_local.dtype)
                weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                (F.log_softmax(pred_corners / T, dim=1), F.softmax(target_corners.detach() / T, dim=1))).sum(-1)
                
                if 'is_dn' not in outputs:
                    batch_scale = 8 / outputs['pred_boxes'].shape[0]
                    self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    
                loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    # ========== 辅助函数 ==========
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """获取跨层的匹配联合集"""
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets = {}
        self.own_targets = {}
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, branch_key='combined', **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': lambda o, t, i, n, **kw: self.loss_local(o, t, i, n, branch_key=branch_key, **kw),
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if len(target_boxes) == 0:
            return {}

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def _compute_branch_loss(self, 
                             branch_outputs: Dict,
                             targets: List[Dict],
                             aux_outputs: List[Dict],
                             pre_outputs: Optional[Dict],
                             dn_outputs: Optional[List],
                             dn_pre_outputs: Optional[Dict],
                             dn_meta: Optional[Dict],
                             branch_key: str,
                             epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        计算单个分支的所有损失
        """
        losses = {}
        device = branch_outputs['pred_logits'].device
        
        # 检查是否有有效的targets
        total_targets = sum(len(t['labels']) for t in targets)
        
        # 🔥 修复：即使 total_targets == 0，也需要参与 all_reduce 以避免死锁
        # 先计算本地 total_targets，然后同步
        total_targets_tensor = torch.as_tensor([total_targets], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(total_targets_tensor)
        global_total_targets = int(total_targets_tensor.item())
        
        # 如果全局都没有目标（所有 GPU 都没有），返回零损失
        # 🔥 修复：必须返回完整的损失字典（包括辅助层），否则会导致 all_reduce 死锁
        if global_total_targets == 0 and branch_key != 'combined':
            # 主损失
            for loss_name in self.losses:
                if loss_name == 'boxes':
                    losses[f'loss_bbox_{branch_key}'] = torch.tensor(0.0, device=device)
                    losses[f'loss_giou_{branch_key}'] = torch.tensor(0.0, device=device)
                elif loss_name == 'local':
                    losses[f'loss_fgl_{branch_key}'] = torch.tensor(0.0, device=device)
                else:
                    losses[f'loss_{loss_name}_{branch_key}'] = torch.tensor(0.0, device=device)
            
            # 辅助层损失（需要与有目标时的 keys 数量一致）
            if aux_outputs:
                for i in range(len(aux_outputs)):
                    for loss_name in self.losses:
                        if loss_name == 'boxes':
                            losses[f'loss_bbox_{branch_key}_aux_{i}'] = torch.tensor(0.0, device=device)
                            losses[f'loss_giou_{branch_key}_aux_{i}'] = torch.tensor(0.0, device=device)
                        elif loss_name == 'local':
                            losses[f'loss_fgl_{branch_key}_aux_{i}'] = torch.tensor(0.0, device=device)
                        else:
                            losses[f'loss_{loss_name}_{branch_key}_aux_{i}'] = torch.tensor(0.0, device=device)
            
            # Pre输出损失
            if pre_outputs is not None:
                for loss_name in self.losses:
                    if loss_name == 'boxes':
                        losses[f'loss_bbox_{branch_key}_pre'] = torch.tensor(0.0, device=device)
                        losses[f'loss_giou_{branch_key}_pre'] = torch.tensor(0.0, device=device)
                    elif loss_name == 'local':
                        # 🔥 修复：pre 输出也包含 local 损失
                        losses[f'loss_fgl_{branch_key}_pre'] = torch.tensor(0.0, device=device)
                    else:
                        losses[f'loss_{loss_name}_{branch_key}_pre'] = torch.tensor(0.0, device=device)
            
            return losses
        
        # 主输出匹配（即使本地没有目标，也需要创建空的 indices）
        if total_targets > 0:
            indices = self.matcher(branch_outputs, targets, epoch=epoch)['indices']
        else:
            # 本地没有目标，创建空的 indices
            batch_size = branch_outputs['pred_logits'].shape[0]
            indices = [(torch.tensor([], dtype=torch.long, device=device), 
                       torch.tensor([], dtype=torch.long, device=device)) 
                      for _ in range(batch_size)]
        
        # 辅助层匹配
        indices_aux_list = []
        cached_indices = []
        
        if aux_outputs:
            aux_list = aux_outputs.copy()
            if pre_outputs is not None:
                aux_list = aux_list + [pre_outputs]
            
            for aux_out in aux_list:
                if total_targets > 0:
                    indices_aux = self.matcher(aux_out, targets, epoch=epoch)['indices']
                else:
                    batch_size = aux_out['pred_logits'].shape[0]
                    indices_aux = [(torch.tensor([], dtype=torch.long, device=device), 
                                   torch.tensor([], dtype=torch.long, device=device)) 
                                  for _ in range(batch_size)]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
        
        # 计算统一集合
        # 🔥 修复：始终执行 all_reduce，即使本地 num_boxes_go 为 0
        if indices_aux_list:
            indices_go = self._get_go_indices(indices, indices_aux_list)
            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            indices_go = indices
            num_boxes_go = max(global_total_targets / get_world_size() if is_dist_available_and_initialized() else total_targets, 1)
        
        # 归一化因子（使用本地 total_targets，通过 all_reduce 同步）
        num_boxes = torch.as_tensor([total_targets], dtype=torch.float, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # 计算主损失
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            
            meta = self.get_loss_meta_info(loss, branch_outputs, targets, indices_in)
            l_dict = self.get_loss(loss, branch_outputs, targets, indices_in, num_boxes_in, 
                                   branch_key=branch_key, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0) for k in l_dict if k in self.weight_dict}
            l_dict = {f'{k}_{branch_key}': v for k, v in l_dict.items()}
            losses.update(l_dict)
        
        # 辅助层损失
        if aux_outputs:
            for i, aux_out in enumerate(aux_outputs):
                if 'local' in self.losses and 'up' in branch_outputs:
                    aux_out['up'] = branch_outputs['up']
                    aux_out['reg_scale'] = branch_outputs['reg_scale']
                
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    
                    meta = self.get_loss_meta_info(loss, aux_out, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_out, targets, indices_in, num_boxes_in,
                                           branch_key=branch_key, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0) for k in l_dict if k in self.weight_dict}
                    l_dict = {f'{k}_{branch_key}_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        # Pre输出损失
        if pre_outputs is not None:
            for loss in self.losses:
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                
                meta = self.get_loss_meta_info(loss, pre_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, pre_outputs, targets, indices_in, num_boxes_in,
                                       branch_key=branch_key, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0) for k in l_dict if k in self.weight_dict}
                l_dict = {f'{k}_{branch_key}_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        # 去噪损失
        # 注意：dn_meta 是基于原始 targets 生成的，但 individual/colony 分支使用的是分割后的 targets
        # 对于 individual/colony 分支，由于 targets 被分割，dn_meta 的索引会不匹配
        # 因此只对 combined 分支计算去噪损失（combined 分支使用完整的 targets）
        if dn_outputs is not None and dn_meta is not None and branch_key == 'combined':
            # 只有 combined 分支使用完整的 targets，可以安全地计算去噪损失
            indices_dn = self.get_cdn_matched_indices(dn_meta, targets)
            dn_num_boxes = num_boxes * dn_meta['dn_num_group']
            
            for i, aux_out in enumerate(dn_outputs):
                if 'local' in self.losses and 'up' in branch_outputs:
                    aux_out['is_dn'] = True
                    aux_out['up'] = branch_outputs['up']
                    aux_out['reg_scale'] = branch_outputs['reg_scale']
                
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_out, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_out, targets, indices_dn, dn_num_boxes,
                                           branch_key=branch_key, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0) for k in l_dict if k in self.weight_dict}
                    l_dict = {f'{k}_{branch_key}_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            if dn_pre_outputs is not None:
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, dn_pre_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, dn_pre_outputs, targets, indices_dn, dn_num_boxes,
                                           branch_key=branch_key, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0) for k in l_dict if k in self.weight_dict}
                    l_dict = {f'{k}_{branch_key}_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        # 🔥 确保所有必需的 keys 都存在（避免 all_reduce 死锁）
        num_aux = len(aux_outputs) if aux_outputs else 0
        has_pre = pre_outputs is not None
        num_dn = len(dn_outputs) if dn_outputs is not None else 0
        has_dn_pre = dn_pre_outputs is not None
        
        all_keys = self._get_all_loss_keys(branch_key, num_aux, has_pre, num_dn, has_dn_pre)
        for key in all_keys:
            if key not in losses:
                losses[key] = torch.tensor(0.0, device=device)
        
        return losses

    def forward(self, outputs, targets, epoch=0, **kwargs):
        """
        计算三分支的总损失
        
        Args:
            outputs: 模型输出，包含三个分支的预测
            targets: GT目标列表
            epoch: 当前epoch
        """
        self._clear_cache()
        
        # 根据尺寸分割targets
        targets_small, targets_large = self._split_targets_by_size(targets)
        
        all_losses = {}
        device = outputs['pred_logits_combined'].device if 'pred_logits_combined' in outputs else \
                 outputs['pred_logits_individual'].device if 'pred_logits_individual' in outputs else \
                 outputs['pred_logits_colony'].device
        
        # 🔥 预先计算所有分支的结构信息（用于生成完整的 keys）
        branch_info = {}
        for branch in ['individual', 'colony', 'combined']:
            if f'pred_logits_{branch}' in outputs:
                aux = self._get_branch_aux_outputs(outputs, branch)
                pre = self._get_branch_pre_outputs(outputs, branch)
                dn, dn_pre, _ = self._get_branch_dn_outputs(outputs, branch)
                branch_info[branch] = {
                    'num_aux': len(aux) if aux else 0,
                    'has_pre': pre is not None,
                    'num_dn': len(dn) if dn is not None else 0,
                    'has_dn_pre': dn_pre is not None,
                }
        
        # 1. Individual分支损失（小目标）
        if 'pred_logits_individual' in outputs:
            ind_outputs = self._get_branch_outputs(outputs, 'individual')
            ind_aux = self._get_branch_aux_outputs(outputs, 'individual')
            ind_pre = self._get_branch_pre_outputs(outputs, 'individual')
            ind_dn, ind_dn_pre, ind_dn_meta = self._get_branch_dn_outputs(outputs, 'individual')
            
            ind_losses = self._compute_branch_loss(
                ind_outputs, targets_small, ind_aux, ind_pre,
                ind_dn, ind_dn_pre, ind_dn_meta,
                'individual', epoch
            )
            # 应用分支权重
            ind_losses = {k: v * self.branch_weights['individual'] for k, v in ind_losses.items()}
            all_losses.update(ind_losses)
        
        # 2. Colony分支损失（大目标）
        if 'pred_logits_colony' in outputs:
            col_outputs = self._get_branch_outputs(outputs, 'colony')
            col_aux = self._get_branch_aux_outputs(outputs, 'colony')
            col_pre = self._get_branch_pre_outputs(outputs, 'colony')
            col_dn, col_dn_pre, col_dn_meta = self._get_branch_dn_outputs(outputs, 'colony')
            
            col_losses = self._compute_branch_loss(
                col_outputs, targets_large, col_aux, col_pre,
                col_dn, col_dn_pre, col_dn_meta,
                'colony', epoch
            )
            col_losses = {k: v * self.branch_weights['colony'] for k, v in col_losses.items()}
            all_losses.update(col_losses)
        
        # 3. Combined分支损失（所有目标）
        if 'pred_logits_combined' in outputs:
            comb_outputs = self._get_branch_outputs(outputs, 'combined')
            comb_aux = self._get_branch_aux_outputs(outputs, 'combined')
            comb_pre = self._get_branch_pre_outputs(outputs, 'combined')
            comb_dn, comb_dn_pre, comb_dn_meta = self._get_branch_dn_outputs(outputs, 'combined')
            
            comb_losses = self._compute_branch_loss(
                comb_outputs, targets, comb_aux, comb_pre,
                comb_dn, comb_dn_pre, comb_dn_meta,
                'combined', epoch
            )
            comb_losses = {k: v * self.branch_weights['combined'] for k, v in comb_losses.items()}
            all_losses.update(comb_losses)
        
        # 🔥 确保所有分支的所有 keys 都存在（避免 DDP all_reduce 死锁）
        for branch, info in branch_info.items():
            all_keys = self._get_all_loss_keys(
                branch, 
                info['num_aux'], 
                info['has_pre'], 
                info['num_dn'], 
                info['has_dn_pre']
            )
            for key in all_keys:
                if key not in all_losses:
                    all_losses[key] = torch.tensor(0.0, device=device)
        
        # 数值稳定化
        all_losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in all_losses.items()}
        
        return all_losses

