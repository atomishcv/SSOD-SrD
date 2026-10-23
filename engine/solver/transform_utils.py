"""
几何变换工具：用于将伪标签从 weak image 坐标系变换到 strong image 坐标系

主要功能：
1. 计算从 weak 到 strong 的变换矩阵
2. 将 bounding boxes 从 weak 坐标系变换到 strong 坐标系
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


def compute_weak_to_strong_transform(
    transform_weak: Dict,
    transform_strong: Dict
) -> torch.Tensor:
    """
    计算从 weak image 坐标系到 strong image 坐标系的变换矩阵
    
    Args:
        transform_weak: weak 增强的变换信息
            - 'matrix': 3x3 齐次变换矩阵（从原图到weak）
            - 'scale': (scale_x, scale_y) 缩放因子
            - 'flip': bool 是否翻转
            - 'translate': (dx, dy) 平移量
        transform_strong: strong 增强的变换信息（格式同 transform_weak）
    
    Returns:
        M_weak_to_strong: 3x3 齐次变换矩阵（从 weak 到 strong）
    """
    M_weak = transform_weak.get('matrix')
    M_strong = transform_strong.get('matrix')
    
    if M_weak is None or M_strong is None:
        # 如果没有矩阵，使用 scale, flip, translate 重建
        M_weak = _build_transform_matrix(transform_weak)
        M_strong = _build_transform_matrix(transform_strong)
    
    # 从 weak 到原始：M_weak^(-1)
    # 从原始到 strong：M_strong
    # 组合：M_strong @ M_weak^(-1)
    M_weak_inv = torch.inverse(M_weak)
    M_weak_to_strong = M_strong @ M_weak_inv
    
    return M_weak_to_strong


def _build_transform_matrix(transform: Dict) -> torch.Tensor:
    """
    从 scale, flip, translate 信息重建变换矩阵
    
    Args:
        transform: 包含 'scale', 'flip', 'translate' 的字典
    
    Returns:
        3x3 齐次变换矩阵
    """
    scale = transform.get('scale', (1.0, 1.0))
    if isinstance(scale, tuple):
        scale_x, scale_y = scale
    else:
        scale_x = scale_y = scale
    
    flip = transform.get('flip', False)
    translate = transform.get('translate', (0, 0))
    dx, dy = translate
    
    # 构建变换矩阵（按顺序：缩放 -> 翻转 -> 平移）
    M = torch.eye(3, dtype=torch.float32)
    
    # 缩放
    M[0, 0] = scale_x
    M[1, 1] = scale_y
    
    # 翻转（如果需要，假设是水平翻转）
    if flip:
        # 翻转矩阵：x' = W - x，需要知道图像宽度
        # 这里先不处理，因为需要图像尺寸信息
        # 在实际使用时，如果 flip=True，会在变换 boxes 时单独处理
        pass
    
    # 平移
    M[0, 2] = dx
    M[1, 2] = dy
    
    return M


def transform_boxes_weak_to_strong(
    boxes_weak: torch.Tensor,
    transform_weak: Dict,
    transform_strong: Dict,
    img_size_weak: Tuple[int, int],
    img_size_strong: Tuple[int, int]
) -> torch.Tensor:
    """
    将 bounding boxes 从 weak image 坐标系变换到 strong image 坐标系
    
    Args:
        boxes_weak: [N, 4] tensor，格式为 (x1, y1, x2, y2) 或 (cx, cy, w, h)
        transform_weak: weak 增强的变换信息
        transform_strong: strong 增强的变换信息
        img_size_weak: (H, W) weak image 尺寸
        img_size_strong: (H, W) strong image 尺寸
    
    Returns:
        boxes_strong: [N, 4] tensor，变换后的 boxes
    """
    # 计算变换矩阵
    M_weak_to_strong = compute_weak_to_strong_transform(transform_weak, transform_strong)
    
    # 将 boxes 转换为齐次坐标
    # boxes 格式：假设是 (x1, y1, x2, y2)
    N = boxes_weak.shape[0]
    if N == 0:
        return boxes_weak.clone()
    
    # 提取四个角点
    x1, y1, x2, y2 = boxes_weak[:, 0], boxes_weak[:, 1], boxes_weak[:, 2], boxes_weak[:, 3]
    
    # 四个角点
    corners = torch.stack([
        torch.stack([x1, y1], dim=1),  # 左上
        torch.stack([x2, y1], dim=1),  # 右上
        torch.stack([x2, y2], dim=1),  # 右下
        torch.stack([x1, y2], dim=1),  # 左下
    ], dim=1)  # [N, 4, 2]
    
    # 转换为齐次坐标 [N, 4, 3]
    ones = torch.ones(N, 4, 1, device=boxes_weak.device, dtype=boxes_weak.dtype)
    corners_homo = torch.cat([corners, ones], dim=2)  # [N, 4, 3]
    
    # 应用变换矩阵
    # M_weak_to_strong: [3, 3]
    # corners_homo: [N, 4, 3]
    # 需要：corners_homo @ M_weak_to_strong.T
    M = M_weak_to_strong.to(boxes_weak.device)
    corners_transformed = corners_homo @ M.T  # [N, 4, 3]
    
    # 提取 x, y 坐标
    corners_transformed_xy = corners_transformed[:, :, :2]  # [N, 4, 2]
    
    # 计算新的 bounding box（取最小外接矩形）
    x_min = corners_transformed_xy[:, :, 0].min(dim=1)[0]  # [N]
    y_min = corners_transformed_xy[:, :, 1].min(dim=1)[0]  # [N]
    x_max = corners_transformed_xy[:, :, 0].max(dim=1)[0]  # [N]
    y_max = corners_transformed_xy[:, :, 1].max(dim=1)[0]  # [N]
    
    # 裁剪到图像边界
    x_min = torch.clamp(x_min, 0, img_size_strong[1] - 1)
    y_min = torch.clamp(y_min, 0, img_size_strong[0] - 1)
    x_max = torch.clamp(x_max, 0, img_size_strong[1] - 1)
    y_max = torch.clamp(y_max, 0, img_size_strong[0] - 1)
    
    # 组合为 boxes
    boxes_strong = torch.stack([x_min, y_min, x_max, y_max], dim=1)  # [N, 4]
    
    # 🔥 不过滤无效 boxes，保持固定数量（DETR 需要固定数量的 queries）
    # 无效的 boxes 会在后续的损失计算中被忽略（通过匹配器）
    # 如果需要过滤，可以在调用时处理
    return boxes_strong


def transform_boxes_cxcywh_weak_to_strong(
    boxes_weak: torch.Tensor,
    transform_weak: Dict,
    transform_strong: Dict,
    img_size_weak: Tuple[int, int],
    img_size_strong: Tuple[int, int]
) -> torch.Tensor:
    """
    将 bounding boxes（cxcywh 格式）从 weak image 坐标系变换到 strong image 坐标系
    
    Args:
        boxes_weak: [N, 4] tensor，格式为 (cx, cy, w, h)
        transform_weak: weak 增强的变换信息
        transform_strong: strong 增强的变换信息
        img_size_weak: (H, W) weak image 尺寸
        img_size_strong: (H, W) strong image 尺寸
    
    Returns:
        boxes_strong: [N, 4] tensor，格式为 (cx, cy, w, h)
    """
    # 转换为 xyxy 格式
    cx, cy, w, h = boxes_weak[:, 0], boxes_weak[:, 1], boxes_weak[:, 2], boxes_weak[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    
    # 变换
    boxes_xyxy_strong = transform_boxes_weak_to_strong(
        boxes_xyxy, transform_weak, transform_strong, img_size_weak, img_size_strong
    )
    
    # 转换回 cxcywh 格式
    x1, y1, x2, y2 = boxes_xyxy_strong[:, 0], boxes_xyxy_strong[:, 1], boxes_xyxy_strong[:, 2], boxes_xyxy_strong[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    
    boxes_strong = torch.stack([cx, cy, w, h], dim=1)
    
    return boxes_strong

