#!/usr/bin/env python3
"""
评估脚本：在新标注数据集上评估模型精度

功能：
1. 加载模型（输出两个类别：small 和 big）
2. 合并两个类别为单一类别
3. 使用跨类别 NMS 去除重复框
4. 与单类别真值进行精度评估

使用方法:
    cd /data/DEIMv2
    conda activate deimv2
    python labeled_folders/evaluate_model.py
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from faster_coco_eval import COCO, COCOeval_faster
from torchvision.ops import nms


def load_model_from_checkpoint(checkpoint_path: str, cfg, device: torch.device):
    """从检查点加载模型"""
    print(f"\n{'='*80}")
    print("📦 加载模型...")
    print(f"{'='*80}")
    print(f"检查点: {checkpoint_path}")
    print(f"设备: {device}\n")
    
    # 创建模型
    model = cfg.model.to(device)
    model.eval()
    
    # 加载权重
    print("加载检查点权重...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 处理不同的检查点格式
    if 'ema' in checkpoint:
        ema_state = checkpoint['ema']
        if isinstance(ema_state, dict) and 'module' in ema_state:
            state_dict = ema_state['module']
            print("✅ 使用 EMA 模型权重 (ema.module)")
        else:
            state_dict = ema_state
            print("✅ 使用 EMA 模型权重 (ema)")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("✅ 使用 'model' 键中的权重")
    else:
        state_dict = checkpoint
        print("✅ 直接加载 state_dict")
    
    # 移除 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"⚠️  缺失的键: {len(missing_keys)} 个")
    if unexpected_keys:
        print(f"⚠️  意外的键: {len(unexpected_keys)} 个")
    
    # 设置动态位置编码
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'eval_spatial_size'):
        model.encoder.eval_spatial_size = None
        print("✅ 已设置动态位置编码")
    
    print(f"✅ 模型加载完成\n")
    return model


def load_coco_gt(ann_file: str):
    """加载 COCO GT 对象"""
    print(f"📂 加载标注文件: {ann_file}")
    
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"标注文件不存在: {ann_file}")
    
    coco_gt = COCO(ann_file)
    
    print(f"✅ 标注文件加载完成")
    print(f"   - 图像数量: {len(coco_gt.imgs)}")
    print(f"   - 标注数量: {len(coco_gt.anns)}")
    print(f"   - 类别: {[cat['name'] for cat in coco_gt.cats.values()]}\n")
    
    return coco_gt


def merge_and_nms(results, nms_iou_threshold=0.5, containment_threshold=0.8):
    """
    合并类别并进行跨类别 NMS
    
    Args:
        results: 模型预测结果列表，每个元素包含 boxes, scores, labels
        nms_iou_threshold: NMS IoU 阈值
        containment_threshold: 嵌套框过滤阈值
    
    Returns:
        merged_results: 合并后的结果
    """
    merged_results = []
    
    for result in results:
        boxes = result['boxes']  # [N, 4] xyxy 格式
        scores = result['scores']  # [N]
        labels = result['labels']  # [N]
        
        if len(boxes) == 0:
            merged_results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': torch.zeros_like(labels)  # 合并为类别 0
            })
            continue
        
        # 1. 跨类别 NMS：不区分类别，所有重叠框都会被去除
        keep_indices = nms(boxes, scores, nms_iou_threshold)
        
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        
        # 2. 嵌套框过滤：去除被大框包含的小框
        if len(boxes) > 1:
            keep_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
            
            for i in range(len(boxes)):
                if not keep_mask[i]:
                    continue
                    
                box_i = boxes[i]
                area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                
                for j in range(len(boxes)):
                    if i == j or not keep_mask[j]:
                        continue
                    
                    box_j = boxes[j]
                    area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                    
                    # 计算交集
                    inter_x1 = max(box_i[0], box_j[0])
                    inter_y1 = max(box_i[1], box_j[1])
                    inter_x2 = min(box_i[2], box_j[2])
                    inter_y2 = min(box_i[3], box_j[3])
                    
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        
                        # 检查小框是否被大框包含
                        if area_i < area_j:
                            containment = inter_area / area_i
                            if containment > containment_threshold:
                                # 小框被大框包含，保留置信度高的
                                if scores[i] < scores[j]:
                                    keep_mask[i] = False
                        else:
                            containment = inter_area / area_j
                            if containment > containment_threshold:
                                if scores[j] < scores[i]:
                                    keep_mask[j] = False
            
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            labels = labels[keep_mask]
        
        # 3. 合并所有类别为单一类别 (category_id=0)
        merged_labels = torch.zeros_like(labels)
        
        merged_results.append({
            'boxes': boxes,
            'scores': scores,
            'labels': merged_labels
        })
    
    return merged_results


def evaluate(
    model, postprocessor, data_loader, coco_gt, device,
    conf_thresh=0.3, nms_iou_threshold=0.5, use_hsv=True
):
    """
    执行评估
    
    Args:
        model: 模型
        postprocessor: 后处理器
        data_loader: 数据加载器
        coco_gt: COCO GT 对象
        device: 设备
        conf_thresh: 置信度阈值
        nms_iou_threshold: NMS IoU 阈值
        use_hsv: 是否使用 HSV 输入
    
    Returns:
        stats: 评估统计
    """
    model.eval()
    
    print(f"\n{'='*80}")
    print("🚀 开始评估...")
    print(f"{'='*80}")
    print(f"置信度阈值: {conf_thresh}")
    print(f"NMS IoU 阈值: {nms_iou_threshold}")
    print(f"使用 HSV: {use_hsv}\n")
    
    # 收集所有预测结果
    all_predictions = []
    
    # 统计信息
    total_before_filter = 0
    total_after_filter = 0
    total_after_nms = 0
    class_0_count = 0
    class_1_count = 0
    
    for batch_idx, (samples, targets) in enumerate(data_loader):
        # 处理输入
        hsv_samples = None
        if isinstance(samples, dict):
            rgb_samples = samples['rgb'].to(device)
            hsv_samples = samples['hsv'].to(device)
            samples = rgb_samples
        else:
            samples = samples.to(device)
        
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in t.items()} for t in targets]
        
        # 前向推理
        with torch.no_grad():
            outputs = model(samples, hsv=hsv_samples)
        
        # 检查 NaN/Inf
        if 'pred_boxes' in outputs:
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                outputs['pred_boxes'] = torch.nan_to_num(outputs['pred_boxes'], nan=0.5, posinf=1.0, neginf=0.0)
        
        if 'pred_logits' in outputs:
            if torch.isnan(outputs['pred_logits']).any() or torch.isinf(outputs['pred_logits']).any():
                outputs['pred_logits'] = torch.nan_to_num(outputs['pred_logits'], nan=0.0, posinf=1e4, neginf=-1e4)
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        # 后处理
        results = postprocessor(outputs, orig_target_sizes)
        
        # 统计置信度过滤前
        for result in results:
            total_before_filter += len(result['labels'])
            if len(result['labels']) > 0:
                class_0_count += (result['labels'] == 0).sum().item()
                class_1_count += (result['labels'] == 1).sum().item()
        
        # 置信度过滤
        if conf_thresh > 0:
            filtered_results = []
            for result in results:
                if len(result['scores']) > 0:
                    keep_mask = result['scores'] >= conf_thresh
                    filtered_results.append({
                        'boxes': result['boxes'][keep_mask],
                        'scores': result['scores'][keep_mask],
                        'labels': result['labels'][keep_mask],
                    })
                else:
                    filtered_results.append(result)
            results = filtered_results
        
        # 统计置信度过滤后
        for result in results:
            total_after_filter += len(result['labels'])
        
        # 合并类别 + NMS
        results = merge_and_nms(results, nms_iou_threshold)
        
        # 统计 NMS 后
        for result in results:
            total_after_nms += len(result['labels'])
        
        # 收集预测结果（COCO 格式）
        for target, result in zip(targets, results):
            image_id = target['image_id'].item()
            
            boxes = result['boxes'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            labels = result['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                # COCO 格式：[x, y, width, height]
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                all_predictions.append({
                    'image_id': image_id,
                    'category_id': 1,  # 真值使用 category_id=1 (plant)
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score)
                })
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  处理进度: {batch_idx + 1}/{len(data_loader)} batches")
    
    # 打印统计信息
    print(f"\n📊 预测统计:")
    print(f"   - 置信度过滤前: {total_before_filter} 个框")
    print(f"     - 类别 0 (small): {class_0_count}")
    print(f"     - 类别 1 (big): {class_1_count}")
    print(f"   - 置信度过滤后: {total_after_filter} 个框")
    print(f"   - NMS + 合并后: {total_after_nms} 个框")
    print(f"   - 最终预测数: {len(all_predictions)} 个框")
    
    # 使用 COCO API 评估
    print(f"\n{'='*80}")
    print("📊 COCO 评估...")
    print(f"{'='*80}\n")
    
    if len(all_predictions) == 0:
        print("⚠️  没有预测结果！")
        return None
    
    # 创建 COCO 预测结果对象
    coco_dt = coco_gt.loadRes(all_predictions)
    
    # 创建评估器
    coco_eval = COCOeval_faster(coco_gt, coco_dt, 'bbox')
    
    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 提取统计信息
    stats = coco_eval.stats
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='在新标注数据集上评估模型精度')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/deimv2_dinov3_ssod/checkpoint_best_ap0.9305_epoch189.pth',
        help='模型检查点路径'
    )
    parser.add_argument(
        '--ann-file',
        type=str,
        default='labeled_folders/annotations/val_labeled.json',
        help='验证集标注文件路径'
    )
    parser.add_argument(
        '--img-folder',
        type=str,
        default='labeled_folders/images',
        help='图像文件夹路径'
    )
    parser.add_argument(
        '--conf-thresh',
        type=float,
        default=0.2,
        help='置信度阈值（默认: 0.3）'
    )
    parser.add_argument(
        '--nms-iou',
        type=float,
        default=0.5,
        help='NMS IoU 阈值（默认: 0.5）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='设备'
    )
    parser.add_argument(
        '--use-hsv',
        action='store_true',
        default=True,
        help='是否使用 HSV 输入'
    )
    
    args = parser.parse_args()
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    # 检查文件
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    if not os.path.exists(args.ann_file):
        raise FileNotFoundError(f"标注文件不存在: {args.ann_file}")
    
    # 设置设备
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，使用 CPU")
        device = torch.device('cpu')
    
    # 初始化分布式
    dist_utils.setup_distributed(print_rank=0, print_method='builtin', seed=0)
    
    # 加载配置
    print(f"\n{'='*80}")
    print("📋 加载配置...")
    print(f"{'='*80}")
    print(f"配置文件: {args.config}\n")
    
    cfg = YAMLConfig(args.config, resume=None)
    
    # 修改验证数据加载器配置
    ann_file_abs = os.path.abspath(args.ann_file)
    img_folder_abs = os.path.abspath(args.img_folder)
    
    if hasattr(cfg, 'yaml_cfg') and 'val_dataloader' in cfg.yaml_cfg:
        cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = ann_file_abs
        cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'] = img_folder_abs
        print(f"✅ 已设置验证集标注文件: {ann_file_abs}")
        print(f"✅ 已设置图像文件夹: {img_folder_abs}")
    
    # 加载模型
    model = load_model_from_checkpoint(args.checkpoint, cfg, device)
    
    # 加载 COCO GT
    coco_gt = load_coco_gt(args.ann_file)
    
    # 创建数据加载器
    print(f"\n{'='*80}")
    print("📊 创建验证数据加载器...")
    print(f"{'='*80}\n")
    
    val_dataloader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    print(f"✅ 验证数据加载器创建完成")
    print(f"   - 数据集大小: {len(val_dataloader.dataset)}\n")
    
    # 获取后处理器
    postprocessor = cfg.postprocessor
    
    # 执行评估
    stats = evaluate(
        model=model,
        postprocessor=postprocessor,
        data_loader=val_dataloader,
        coco_gt=coco_gt,
        device=device,
        conf_thresh=args.conf_thresh,
        nms_iou_threshold=args.nms_iou,
        use_hsv=args.use_hsv
    )
    
    # 打印结果
    if stats is not None:
        print(f"\n{'='*80}")
        print("📊 评估结果")
        print(f"{'='*80}")
        print(f"\nCOCO 评估指标 (合并类别):")
        print(f"  AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]:.4f}")
        print(f"  AP @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]:.4f}")
        print(f"  AP @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]:.4f}")
        print(f"  AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]:.4f}")
        print(f"  AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]:.4f}")
        print(f"  AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]:.4f}")
        print(f"  AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats[6]:.4f}")
        print(f"  AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats[7]:.4f}")
        print(f"  AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]:.4f}")
        print(f"  AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[9]:.4f}")
        print(f"  AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[10]:.4f}")
        print(f"  AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[11]:.4f}")
        
        print(f"\n🎯 主要指标:")
        print(f"  mAP@[0.5:0.95]: {stats[0]:.4f}")
        print(f"  AP@0.5:         {stats[1]:.4f}")
        print(f"  AP@0.75:        {stats[2]:.4f}")
    
    print(f"\n{'='*80}")
    print("✅ 评估完成")
    print(f"{'='*80}\n")
    
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
