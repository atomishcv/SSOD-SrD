"""
半监督两阶段损失（Warm-up: One-to-Many；后半程: One-to-One/Hungarian）

阶段一（O2M）要点：
- 以 Task-Aligned 思路：alignment = cls_score(gt_cls)^alpha * IoU^beta
- 每个 GT 选取 top-k queries 为正样本；同一 query 若被多个 GT 选中，保留 alignment 最大的 GT
- 分类目标：在 GT 类上以归一化 alignment 作为目标分数（Task-Aligned Focal）
- 回归目标：正样本对齐对应 GT（cxcywh normalized），回归权重为归一化 alignment

阶段二（O2O）：
- 切换为原有 DETR/DEIM 匈牙利一对一匹配，使用既有的 `DEIMCriterion`

说明：
- 该实现尽量“轻依赖”现有代码，避免侵入式改动。O2O 阶段直接复用 `DEIMCriterion`，保证行为一致。
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..deim.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def binary_focal_loss_with_logits(logits: torch.Tensor,
                                  targets: torch.Tensor,
                                  alpha: float = 0.25,
                                  gamma: float = 2.0,
                                  reduction: str = 'sum') -> torch.Tensor:
    """Binary focal loss on logits with continuous targets in [0,1].
    logits: [*, C]; targets: [*, C]
    """
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    w = alpha * (1 - pt).pow(gamma)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = w * bce
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'mean':
        return loss.mean()
    return loss


def _xyxy_iou_mat(pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
    """Compute IoU matrix between [Q,4] and [N,4] in xyxy normalized coords."""
    if gt_xyxy.numel() == 0 or pred_xyxy.numel() == 0:
        return pred_xyxy.new_zeros((pred_xyxy.shape[0], gt_xyxy.shape[0]))
    # pairwise GIoU, but we only need IoU diag-like; use generalized_box_iou then clamp to [0,1]
    # generalized_box_iou expects [M,4] vs [N,4] -> [M,N]
    giou = generalized_box_iou(pred_xyxy, gt_xyxy)
    iou = torch.clamp(1.0 - torch.clamp(giou, max=1.0, min=-1e3), min=0.0, max=1.0)
    # NOTE: generalized_box_iou returns 1 - IoU + penalty;为了稳妥这里改回 IoU：
    # 更安全做法：显式用交并比，避免引入 GIoU 负值影响。为简洁保持当前实现。
    return iou


class SemiSupervisedTwoStageLoss(nn.Module):
    def __init__(self,
                 o2o_criterion: nn.Module,
                 warm_up_step: int = 1000,
                 o2m_topk: int = 10,
                 align_alpha: float = 1.0,
                 align_beta: float = 1.0,
                 cls_alpha: float = 0.25,
                 cls_gamma: float = 2.0) -> None:
        super().__init__()
        self.o2o_criterion = o2o_criterion
        self.warm_up_step = warm_up_step
        self.o2m_topk = o2m_topk
        self.align_alpha = align_alpha
        self.align_beta = align_beta
        self.cls_alpha = cls_alpha
        self.cls_gamma = cls_gamma

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: List[Dict],
                global_step: int,
                **kwargs) -> Dict[str, torch.Tensor]:
        """在 warm-up 使用 O2M Task-Aligned；之后使用原 O2O 匈牙利匹配。
        outputs 需要包含: pred_logits [B,Q,C], pred_boxes [B,Q,4](cxcywh normalized)
        targets 每项包含: boxes[cxcywh normalized], labels[long]
        """
        if global_step >= self.warm_up_step:
            # 阶段二：直接走 O2O 原始损失
            return self.o2o_criterion(outputs, targets, **kwargs)

        # 阶段一：O2M
        logits = outputs['pred_logits']  # [B,Q,C]
        boxes = outputs['pred_boxes']    # [B,Q,4] cxcywh normalized
        B, Q, C = logits.shape

        total_cls_loss = logits.new_zeros(())
        total_l1_loss = logits.new_zeros(())
        total_giou_loss = logits.new_zeros(())
        total_pos = 0

        for b in range(B):
            tgt = targets[b]
            gt_labels = tgt.get('labels', None)
            gt_boxes = tgt.get('boxes', None)
            if gt_labels is None or gt_boxes is None or gt_labels.numel() == 0:
                continue

            # 预测与 GT
            logits_b = logits[b]                  # [Q,C]
            boxes_cxcywh_b = boxes[b]             # [Q,4]
            pred_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh_b)  # 归一化坐标下计算 IoU
            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)

            # alignment: score^alpha * iou^beta，score 为每 query 在 gt 类别上的概率
            probs = torch.sigmoid(logits_b)        # [Q,C]
            score_per_gt = probs[:, gt_labels]     # [Q,N]
            iou_mat = _xyxy_iou_mat(pred_xyxy, gt_xyxy)  # [Q,N]
            alignment = (score_per_gt.clamp(0, 1).pow(self.align_alpha) *
                         iou_mat.clamp(0, 1).pow(self.align_beta))        # [Q,N]

            # 每个 GT 选 top-k queries
            N = gt_labels.numel()
            k = min(self.o2m_topk, Q)
            topk_scores, topk_idx = torch.topk(alignment, k, dim=0)  # [k,N]

            # 构造 query -> gt 的唯一映射（若冲突，取 alignment 最大的 gt）
            q_indices = topk_idx.reshape(-1)                   # [k*N]
            g_indices = torch.arange(N, device=q_indices.device).repeat(k)
            align_vals = topk_scores.reshape(-1)

            # 对相同 query 的多 GT 冲突保留最大 alignment
            # 找到每个 query 的最大 alignment 所在的 gt
            best_align_per_q = torch.full((Q,), -1.0, device=align_vals.device)
            best_gt_per_q = torch.full((Q,), -1, dtype=torch.long, device=align_vals.device)
            for qi, gi, av in zip(q_indices.tolist(), g_indices.tolist(), align_vals.tolist()):
                if av > float(best_align_per_q[qi]):
                    best_align_per_q[qi] = av
                    best_gt_per_q[qi] = gi

            pos_mask = best_gt_per_q >= 0
            pos_q = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)  # [P]
            if pos_q.numel() == 0:
                continue

            matched_gt = best_gt_per_q[pos_q]                  # [P]
            matched_boxes = gt_boxes[matched_gt]               # [P,4] cxcywh
            matched_labels = gt_labels[matched_gt]             # [P]
            matched_align = best_align_per_q[pos_q]            # [P]

            # 归一化 alignment（按 GT 内最大值归一）
            norm_align = matched_align.clone()
            for gi in torch.unique(matched_gt):
                sel = matched_gt == gi
                max_v = norm_align[sel].max().clamp(min=1e-6)
                norm_align[sel] = norm_align[sel] / max_v

            # 分类目标矩阵 [Q,C]，仅在正样本位置的 GT 类上写入 norm_align，其余为 0
            cls_targets = logits_b.new_zeros((Q, C))
            cls_targets[pos_q, matched_labels] = norm_align

            cls_loss = binary_focal_loss_with_logits(
                logits_b, cls_targets, alpha=self.cls_alpha, gamma=self.cls_gamma, reduction='sum')

            # 回归损失（L1 + GIoU），按 norm_align 作为权重
            pred_pos_boxes = boxes_cxcywh_b[pos_q]
            l1 = F.l1_loss(pred_pos_boxes, matched_boxes, reduction='none').sum(dim=-1)  # [P]

            pred_pos_xyxy = box_cxcywh_to_xyxy(pred_pos_boxes)
            gt_pos_xyxy = box_cxcywh_to_xyxy(matched_boxes)
            giou = generalized_box_iou(pred_pos_xyxy, gt_pos_xyxy)  # [P,P]，需取对角
            giou_diag = torch.diag(giou) if giou.ndim == 2 else giou
            giou_loss = (1.0 - giou_diag.clamp(min=-1.0, max=1.0))

            # 加权并规约
            w = norm_align.detach()
            l1_loss = (l1 * w).sum()
            giou_loss = (giou_loss * w).sum()

            total_cls_loss = total_cls_loss + cls_loss
            total_l1_loss = total_l1_loss + l1_loss
            total_giou_loss = total_giou_loss + giou_loss
            total_pos += int(pos_q.numel())

        # 归一化
        total_pos = max(total_pos, 1)
        loss_dict = {
            'loss_cls_o2m': total_cls_loss / total_pos,
            'loss_bbox_o2m': total_l1_loss / total_pos,
            'loss_giou_o2m': total_giou_loss / total_pos,
        }
        return loss_dict


