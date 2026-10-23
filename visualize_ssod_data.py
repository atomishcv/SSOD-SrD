"""
可视化 SSOD 训练数据（有标注 + 无标注）
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.core import YAMLConfig
from engine.data.ssod_group_reader import SSODGroupReader
from engine.misc import dist_utils


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

# 为不同类别生成颜色
np.random.seed(42)
COLORS = {}
for i, cls_name in enumerate(COCO_CLASSES):
    COLORS[cls_name] = tuple(np.random.randint(50, 255, 3).tolist())


def denormalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """反归一化图像"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # 如果是 CHW 格式，转换为 HWC
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # 反归一化
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    # BGR to RGB (OpenCV 使用 BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


def draw_boxes_on_image(image, boxes, labels, tag='labeled', scores=None):
    """
    在图像上绘制边界框
    
    Args:
        image: numpy array (H, W, 3) BGR格式
        boxes: tensor or list, (N, 4) 格式为 [cx, cy, w, h] 归一化坐标
        labels: tensor or list, (N,) 类别ID
        tag: str, 'labeled' 或 'unlabeled'
        scores: tensor or list, (N,) 置信度分数（可选）
    """
    image = image.copy()
    h, w = image.shape[:2]
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        # 转换归一化坐标到像素坐标
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        
        # 获取类别名称和颜色
        label_id = int(label)
        if label_id < 0 or label_id >= len(COCO_CLASSES):
            continue
        
        class_name = COCO_CLASSES[label_id]
        color = COLORS[class_name]
        
        # 绘制边界框
        thickness = 2 if tag == 'labeled' else 1
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 准备标签文本
        if scores is not None and idx < len(scores):
            text = f'{class_name}: {scores[idx]:.2f}'
        else:
            text = class_name
        
        # 绘制标签背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        cv2.rectangle(image, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(image, text, (x1, y1 - baseline - 2), font, font_scale, (255, 255, 255), font_thickness)
    
    # 在图像顶部添加标签类型标记
    tag_text = f'[{tag.upper()}]'
    tag_color = (0, 255, 0) if tag == 'labeled' else (255, 165, 0)  # 绿色/橙色
    cv2.putText(image, tag_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tag_color, 2)
    
    return image


def visualize_ssod_batch(labeled_batch, unlabeled_batch_teacher, unlabeled_batch_student, 
                          save_dir, batch_idx, max_samples=4):
    """
    可视化一个 batch 的 SSOD 数据
    
    Args:
        labeled_batch: tuple (images_sup, targets_sup)
        unlabeled_batch_teacher: tuple (images_w, targets_w)
        unlabeled_batch_student: tuple (images_s, targets_s)
        save_dir: str, 保存目录
        batch_idx: int, batch 索引
        max_samples: int, 每个 batch 最多可视化多少张
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    images_sup, targets_sup = labeled_batch
    images_w, targets_w = unlabeled_batch_teacher
    images_s, targets_s = unlabeled_batch_student
    
    batch_size = min(images_sup.shape[0], max_samples)
    
    for i in range(batch_size):
        # 1. 可视化有标注数据
        img_sup = denormalize_image(images_sup[i])
        target_sup = targets_sup[i]
        
        if 'boxes' in target_sup and 'labels' in target_sup:
            boxes_sup = target_sup['boxes']
            labels_sup = target_sup['labels']
            img_sup_with_boxes = draw_boxes_on_image(img_sup, boxes_sup, labels_sup, tag='labeled')
        else:
            img_sup_with_boxes = img_sup
            cv2.putText(img_sup_with_boxes, '[LABELED - No annotations]', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 2. 可视化无标注数据（弱增强 - 教师输入）
        img_w = denormalize_image(images_w[i])
        cv2.putText(img_w, '[UNLABELED - Weak Aug]', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
        
        # 3. 可视化无标注数据（强增强 - 学生输入）
        img_s = denormalize_image(images_s[i])
        cv2.putText(img_s, '[UNLABELED - Strong Aug]', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
        
        # 拼接三张图像（横向）
        # 调整大小以确保高度一致
        h = max(img_sup_with_boxes.shape[0], img_w.shape[0], img_s.shape[0])
        w = max(img_sup_with_boxes.shape[1], img_w.shape[1], img_s.shape[1])
        
        # 创建白色背景
        combined = np.ones((h, w * 3, 3), dtype=np.uint8) * 255
        
        # 放置三张图像
        combined[:img_sup_with_boxes.shape[0], :img_sup_with_boxes.shape[1]] = img_sup_with_boxes
        combined[:img_w.shape[0], w:w+img_w.shape[1]] = img_w
        combined[:img_s.shape[0], 2*w:2*w+img_s.shape[1]] = img_s
        
        # 添加分隔线
        cv2.line(combined, (w, 0), (w, h), (0, 0, 0), 2)
        cv2.line(combined, (2*w, 0), (2*w, h), (0, 0, 0), 2)
        
        # 保存
        save_path = save_dir / f'batch{batch_idx:04d}_sample{i}.jpg'
        cv2.imwrite(str(save_path), combined)
        print(f'✅ 保存: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='可视化 SSOD 训练数据')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--num-batches', type=int, default=5,
                       help='可视化多少个 batch')
    parser.add_argument('--max-samples', type=int, default=4,
                       help='每个 batch 最多可视化多少张图')
    parser.add_argument('--save-dir', type=str, default='./visualizations/ssod_data',
                       help='保存目录')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"📊 SSOD 数据可视化")
    print(f"{'='*80}")
    print(f"配置文件: {args.config}")
    print(f"可视化 batch 数: {args.num_batches}")
    print(f"每 batch 样本数: {args.max_samples}")
    print(f"保存目录: {args.save_dir}")
    print(f"{'='*80}\n")
    
    # 加载配置
    cfg = YAMLConfig(args.config, resume=None)
    
    # 使用已构建的数据加载器
    print("📦 获取数据加载器...")
    
    # cfg 已经构建好了 DataLoader
    labeled_loader = cfg.train_dataloader
    unlabeled_loader = cfg.unlabeled_dataloader
    
    # 创建分组读取器
    group_reader = SSODGroupReader(labeled_loader, unlabeled_loader)
    
    print(f"✅ 数据加载器准备完成")
    print(f"   - 有标注数据集大小: {len(labeled_loader.dataset)}")
    print(f"   - 无标注数据集大小: {len(unlabeled_loader.dataset)}")
    print(f"   - 有标注 batch size: {labeled_loader.batch_size}")
    print(f"   - 无标注 batch size: {unlabeled_loader.batch_size}\n")
    
    # 可视化
    print("🎨 开始可视化...\n")
    
    for batch_idx, groups in enumerate(group_reader):
        if batch_idx >= args.num_batches:
            break
        
        print(f"处理 Batch {batch_idx + 1}/{args.num_batches}...")
        
        labeled_batch = groups['sup']
        unlabeled_batch_teacher = groups['unsup_teacher']
        unlabeled_batch_student = groups['unsup_student']
        
        visualize_ssod_batch(
            labeled_batch,
            unlabeled_batch_teacher,
            unlabeled_batch_student,
            save_dir=args.save_dir,
            batch_idx=batch_idx,
            max_samples=args.max_samples
        )
    
    print(f"\n{'='*80}")
    print(f"✅ 可视化完成！")
    print(f"📁 结果保存在: {args.save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
