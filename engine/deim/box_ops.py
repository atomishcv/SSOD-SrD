"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    # 🔥 保护：清理 NaN/Inf，避免转换时产生无效坐标
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x_c, y_c, w, h = x.unbind(-1)
    # 🔥 确保 w, h >= 0（避免负宽高导致坐标异常）
    w = w.clamp(min=0.0)
    h = h.clamp(min=0.0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    # 🔥 保护：避免 union=0 导致 NaN（当两个框面积都为 0 时）
    iou = inter / union.clamp(min=1e-8)
    # 🔥 清理可能的 NaN/Inf（虽然已经保护，但双重保险）
    iou = torch.nan_to_num(iou, nan=0.0, posinf=1.0, neginf=0.0)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    
    Note: 已移除 assert，改为自动修正非法框以提高鲁棒性
    """
    # ========== 关键修复：预处理框坐标，防止非法值 ==========
    # 1. 清理 NaN/Inf
    boxes1 = torch.nan_to_num(boxes1, nan=0.0, posinf=1e8, neginf=0.0)
    boxes2 = torch.nan_to_num(boxes2, nan=0.0, posinf=1e8, neginf=0.0)
    
    # 2. 修正坐标顺序：确保 x2 >= x1, y2 >= y1
    b1_xymin = torch.minimum(boxes1[:, :2], boxes1[:, 2:])
    b1_xymax = torch.maximum(boxes1[:, :2], boxes1[:, 2:])
    boxes1 = torch.cat([b1_xymin, b1_xymax], dim=-1)
    
    b2_xymin = torch.minimum(boxes2[:, :2], boxes2[:, 2:])
    b2_xymax = torch.maximum(boxes2[:, :2], boxes2[:, 2:])
    boxes2 = torch.cat([b2_xymin, b2_xymax], dim=-1)
    # ========== 预处理完成 ==========
    
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    # 🔥 保护：避免 area=0 导致 NaN
    giou = iou - (area - union) / area.clamp(min=1e-8)
    # 🔥 清理可能的 NaN/Inf
    giou = torch.nan_to_num(giou, nan=0.0, posinf=1.0, neginf=-1.0)
    return giou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)