"""
教师模型伪标签可视化工具

用于半监督目标检测训练过程中，可视化教师模型生成的伪标签。
保存为图像文件供人工检查伪标签质量。
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torchvision.transforms.functional as F


# COCO 类别名称（80类）
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 颜色映射（用于不同类别）
def get_color_map(num_classes: int = 80) -> List[Tuple[int, int, int]]:
    """生成类别颜色映射"""
    np.random.seed(42)  # 固定随机种子以保持颜色一致
    colors = []
    for i in range(num_classes):
        colors.append((
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255))
        ))
    return colors

COLOR_MAP = get_color_map(80)


def denormalize_image(image: torch.Tensor, 
                       mean: List[float] = [0.485, 0.456, 0.406],
                       std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    反标准化图像并转换为 numpy 数组 (BGR, uint8)
    
    Args:
        image: [C, H, W] 标准化后的图像张量
        mean: ImageNet 均值
        std: ImageNet 标准差
    
    Returns:
        [H, W, C] BGR uint8 数组
    """
    # 克隆以避免修改原始数据
    img = image.clone()
    
    # 反标准化
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    
    # 裁剪到 [0, 1]
    img = torch.clamp(img, 0.0, 1.0)
    
    # 转换为 numpy 并调整通道顺序: [C, H, W] -> [H, W, C]
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    
    # 转换为 uint8
    img_np = (img_np * 255).astype(np.uint8)
    
    # RGB -> BGR (OpenCV 使用 BGR)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    return img_np


def draw_boxes_on_image(image: np.ndarray,
                         boxes: torch.Tensor,
                         labels: torch.Tensor,
                         scores: torch.Tensor,
                         score_thresh: float = 0.0,
                         class_names: List[str] = COCO_CLASSES,
                         color_map: List[Tuple[int, int, int]] = COLOR_MAP) -> np.ndarray:
    """
    在图像上绘制边界框和标签
    
    Args:
        image: [H, W, C] BGR uint8 图像
        boxes: [N, 4] 边界框，格式为 [x_center, y_center, width, height]，归一化坐标 [0, 1]
        labels: [N] 类别标签
        scores: [N] 置信度分数
        score_thresh: 分数阈值，低于此值的框不绘制
        class_names: 类别名称列表
        color_map: 颜色映射
    
    Returns:
        绘制了框的图像
    """
    img = image.copy()
    H, W = img.shape[:2]
    
    # 将归一化坐标转换为像素坐标
    boxes_np = boxes.cpu().numpy()
    labels_np = labels.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    for i in range(len(boxes_np)):
        if scores_np[i] < score_thresh:
            continue
        
        # 转换为像素坐标 (中心点格式 -> 左上角格式)
        cx, cy, w, h = boxes_np[i]
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        
        # 裁剪到图像范围
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        
        # 获取类别和颜色
        label_idx = int(labels_np[i])
        if label_idx < 0 or label_idx >= len(class_names):
            continue
        
        class_name = class_names[label_idx]
        color = color_map[label_idx % len(color_map)]
        score = scores_np[i]
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label_text = f"{class_name}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            img,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img


def visualize_pseudo_labels(
    images: torch.Tensor,
    pseudo_labels: List[Dict],
    targets_meta: List[Dict],
    save_dir: str,
    epoch: int,
    step: int,
    max_samples: int = 4,
    score_thresh: float = 0.5,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> None:
    """
    可视化教师模型生成的伪标签
    
    Args:
        images: [B, C, H, W] 图像批次（弱增强后的图像）
        pseudo_labels: 教师模型生成的伪标签列表，每个元素包含 'boxes', 'labels', 'scores'
        targets_meta: 目标元数据列表，用于获取文件名等信息
        save_dir: 保存目录
        epoch: 当前 epoch
        step: 当前 step
        max_samples: 最多可视化的样本数量
        score_thresh: 显示的置信度阈值
        mean: 图像标准化均值
        std: 图像标准化标准差
    """
    # 创建保存目录
    save_path = Path(save_dir) / f"pseudo_labels_epoch{epoch:03d}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 限制可视化数量
    num_samples = min(len(images), len(pseudo_labels), max_samples)
    
    for i in range(num_samples):
        # 反标准化图像
        img_tensor = images[i]  # [C, H, W]
        img_np = denormalize_image(img_tensor, mean, std)
        
        # 获取伪标签
        pseudo = pseudo_labels[i]
        if not isinstance(pseudo, dict):
            continue
        
        boxes = pseudo.get('boxes')
        labels = pseudo.get('labels')
        scores = pseudo.get('scores')
        
        # 检查是否有有效的伪标签
        if boxes is None or labels is None or scores is None:
            continue
        if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
            continue
        
        # 在图像上绘制框
        img_with_boxes = draw_boxes_on_image(
            img_np,
            boxes,
            labels,
            scores,
            score_thresh=score_thresh
        )
        
        # 获取文件名（如果可用）
        file_name = None
        if i < len(targets_meta) and isinstance(targets_meta[i], dict):
            meta = targets_meta[i].get('meta', {})
            file_name = meta.get('file_name', None)
        
        if file_name is None:
            file_name = f"batch{step}_sample{i}"
        else:
            # 提取文件名（去掉路径和扩展名）
            file_name = Path(file_name).stem
        
        # 保存图像
        output_file = save_path / f"step{step:04d}_{file_name}.jpg"
        cv2.imwrite(str(output_file), img_with_boxes)
    
    # 每隔一定 step 打印一次
    if step % 100 == 0:
        print(f"[Pseudo Label Vis] Epoch {epoch}, Step {step}: "
              f"Saved {num_samples} visualizations to {save_path}")


def visualize_teacher_student_comparison(
    images_weak: torch.Tensor,
    images_strong: torch.Tensor,
    teacher_pseudo: List[Dict],
    student_pseudo: List[Dict],
    targets_meta_weak: List[Dict],
    targets_meta_strong: List[Dict],
    save_dir: str,
    epoch: int,
    step: int,
    max_samples: int = 4,
    score_thresh: float = 0.5,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> None:
    """
    并排可视化教师模型（弱增强）和学生模型（强增强）的预测对比
    
    Args:
        images_weak: [B, C, H, W] 弱增强图像
        images_strong: [B, C, H, W] 强增强图像
        teacher_pseudo: 教师模型伪标签
        student_pseudo: 学生模型伪标签
        targets_meta_weak: 弱增强元数据
        targets_meta_strong: 强增强元数据
        save_dir: 保存目录
        epoch: 当前 epoch
        step: 当前 step
        max_samples: 最多可视化的样本数量
        score_thresh: 显示的置信度阈值
        mean: 图像标准化均值
        std: 图像标准化标准差
    """
    # 创建保存目录
    save_path = Path(save_dir) / f"teacher_student_comparison_epoch{epoch:03d}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 限制可视化数量
    num_samples = min(
        len(images_weak), len(images_strong),
        len(teacher_pseudo), len(student_pseudo),
        max_samples
    )
    
    for i in range(num_samples):
        # 反标准化图像
        img_weak_np = denormalize_image(images_weak[i], mean, std)
        img_strong_np = denormalize_image(images_strong[i], mean, std)
        
        # 教师模型伪标签
        teacher = teacher_pseudo[i] if i < len(teacher_pseudo) else {}
        if isinstance(teacher, dict):
            t_boxes = teacher.get('boxes')
            t_labels = teacher.get('labels')
            t_scores = teacher.get('scores')
            
            if (t_boxes is not None and isinstance(t_boxes, torch.Tensor) and 
                t_labels is not None and t_scores is not None):
                img_weak_np = draw_boxes_on_image(
                    img_weak_np, t_boxes, t_labels, t_scores, score_thresh
                )
        
        # 学生模型伪标签
        student = student_pseudo[i] if i < len(student_pseudo) else {}
        if isinstance(student, dict):
            s_boxes = student.get('boxes')
            s_labels = student.get('labels')
            s_scores = student.get('scores')
            
            if (s_boxes is not None and isinstance(s_boxes, torch.Tensor) and 
                s_labels is not None and s_scores is not None):
                img_strong_np = draw_boxes_on_image(
                    img_strong_np, s_boxes, s_labels, s_scores, score_thresh
                )
        
        # 调整图像大小以匹配（如果不同）
        h1, w1 = img_weak_np.shape[:2]
        h2, w2 = img_strong_np.shape[:2]
        target_h = max(h1, h2)
        
        if h1 != target_h:
            scale = target_h / h1
            img_weak_np = cv2.resize(img_weak_np, (int(w1 * scale), target_h))
        if h2 != target_h:
            scale = target_h / h2
            img_strong_np = cv2.resize(img_strong_np, (int(w2 * scale), target_h))
        
        # 添加标题
        title_h = 40
        img_weak_with_title = np.zeros((target_h + title_h, img_weak_np.shape[1], 3), dtype=np.uint8)
        img_weak_with_title[:title_h] = (50, 50, 50)  # 深灰色背景
        img_weak_with_title[title_h:] = img_weak_np
        cv2.putText(
            img_weak_with_title, "Teacher (Weak Aug)",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        img_strong_with_title = np.zeros((target_h + title_h, img_strong_np.shape[1], 3), dtype=np.uint8)
        img_strong_with_title[:title_h] = (50, 50, 50)
        img_strong_with_title[title_h:] = img_strong_np
        cv2.putText(
            img_strong_with_title, "Student (Strong Aug)",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # 横向拼接
        comparison = np.hstack([img_weak_with_title, img_strong_with_title])
        
        # 获取文件名
        file_name = None
        if i < len(targets_meta_weak) and isinstance(targets_meta_weak[i], dict):
            meta = targets_meta_weak[i].get('meta', {})
            file_name = meta.get('file_name', None)
        
        if file_name is None:
            file_name = f"batch{step}_sample{i}"
        else:
            file_name = Path(file_name).stem
        
        # 保存图像
        output_file = save_path / f"step{step:04d}_{file_name}.jpg"
        cv2.imwrite(str(output_file), comparison)
    
    # 每隔一定 step 打印一次
    if step % 100 == 0:
        print(f"[Teacher-Student Vis] Epoch {epoch}, Step {step}: "
              f"Saved {num_samples} comparisons to {save_path}")

