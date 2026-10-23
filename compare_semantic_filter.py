"""
批量测试脚本 - 对比语义过滤前后的差异

遍历 COCO 数据集，找出语义过滤前后检测结果不同的案例，
并输出对比图（左：仅置信度过滤，右：置信度+语义过滤）

使用方式：
    python compare_semantic_filter.py \
        --checkpoint outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth \
        --image-dir /data/usb/coco/train2017/ \
        --output visualizations/semantic_compare/ \
        --max-images 1000
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from engine.core import YAMLConfig

# COCO 类别名称
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

# 用于可视化的颜色（BGR格式）
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128)
]


def load_model(checkpoint_path, config_path, device='cuda'):
    """加载训练好的模型"""
    cfg = YAMLConfig(config_path, resume=None)
    model = cfg.model
    model = model.to(device)
    model.eval()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'ema' in checkpoint:
        ema_state = checkpoint['ema']
        if isinstance(ema_state, dict) and 'module' in ema_state:
            state_dict = ema_state['module']
        else:
            state_dict = ema_state
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'eval_spatial_size'):
        model.encoder.eval_spatial_size = None
    
    return model


def preprocess_image(image_path, input_size=640):
    """预处理图片"""
    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        return None, None, None, None
    
    orig_h, orig_w = orig_image.shape[:2]
    scale = min(input_size / orig_h, input_size / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    
    resized = cv2.resize(orig_image, (new_w, new_h))
    
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    pad_top = (input_size - new_h) // 2
    pad_left = (input_size - new_w) // 2
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    
    image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(pil_image).unsqueeze(0)
    pad_info = (pad_left, pad_top)
    
    return image_tensor, orig_image, scale, pad_info


@torch.no_grad()
def inference_with_two_strategies(model, image_tensor, device, conf_thresh, 
                                  adaptive_proto, proto_percentile, nms_iou):
    """
    使用两种策略进行推理并返回结果
    
    返回:
        pred1: 仅置信度过滤的结果
        pred2: 置信度+语义过滤的结果
    """
    image_tensor = image_tensor.to(device)
    
    # 前向传播
    outputs = model(image_tensor, None, True)
    
    if isinstance(outputs, dict):
        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['pred_logits']
        dec_hs = outputs.get('dec_hs', None)
    else:
        pred_boxes, pred_logits = outputs
        dec_hs = None
    
    pred_scores = pred_logits.sigmoid()
    
    if dec_hs is not None:
        dec_hs = dec_hs[5]
    
    # 获取语义阈值
    proto_thresholds = adaptive_proto.get_adaptive_thresholds(
        percentile=proto_percentile,
        fixed_threshold=None
    ).to(device)
    
    results = []
    
    for i in range(pred_boxes.shape[0]):
        scores, labels = pred_scores[i].max(dim=-1)
        boxes_cxcywh = pred_boxes[i]
        
        # ========== 策略1: 仅置信度过滤 + NMS ==========
        keep1 = scores >= conf_thresh
        if keep1.sum() == 0:
            top_idx = torch.argmax(scores)
            keep1 = torch.zeros_like(scores, dtype=torch.bool)
            keep1[top_idx] = True
        
        boxes1 = boxes_cxcywh[keep1]
        labels1 = labels[keep1]
        scores1 = scores[keep1]
        
        # NMS
        if nms_iou > 0 and len(boxes1) > 0:
            from torchvision.ops import batched_nms
            boxes_xyxy1 = torch.cat([
                boxes1[:, :2] - boxes1[:, 2:] / 2,
                boxes1[:, :2] + boxes1[:, 2:] / 2
            ], dim=-1)
            nms_keep1 = batched_nms(boxes_xyxy1, scores1, labels1, nms_iou)
            boxes1 = boxes1[nms_keep1]
            labels1 = labels1[nms_keep1]
            scores1 = scores1[nms_keep1]
        
        pred1 = {
            'boxes': boxes1.cpu(),
            'labels': labels1.cpu(),
            'scores': scores1.cpu()
        }
        
        # ========== 策略2: 置信度 + 语义过滤 + NMS ==========
        if dec_hs is not None:
            # 置信度预过滤
            conf_keep = scores >= conf_thresh
            if conf_keep.sum() == 0:
                top_idx = torch.argmax(scores)
                conf_keep = torch.zeros_like(scores, dtype=torch.bool)
                conf_keep[top_idx] = True
            
            conf_features = dec_hs[i][conf_keep]
            conf_boxes = boxes_cxcywh[conf_keep]
            conf_labels = labels[conf_keep]
            conf_scores = scores[conf_keep]
            
            # 语义过滤
            proto_similarities = adaptive_proto.compute_similarity(
                conf_features, conf_labels, method='max'
            )
            class_thresholds = proto_thresholds[conf_labels]
            proto_keep = proto_similarities >= class_thresholds
            
            if proto_keep.sum() == 0:
                top_idx = torch.argmax(proto_similarities)
                proto_keep = torch.zeros_like(proto_keep, dtype=torch.bool)
                proto_keep[top_idx] = True
            
            boxes2 = conf_boxes[proto_keep]
            labels2 = conf_labels[proto_keep]
            scores2 = conf_scores[proto_keep]
            
            # NMS
            if nms_iou > 0 and len(boxes2) > 0:
                boxes_xyxy2 = torch.cat([
                    boxes2[:, :2] - boxes2[:, 2:] / 2,
                    boxes2[:, :2] + boxes2[:, 2:] / 2
                ], dim=-1)
                nms_keep2 = batched_nms(boxes_xyxy2, scores2, labels2, nms_iou)
                boxes2 = boxes2[nms_keep2]
                labels2 = labels2[nms_keep2]
                scores2 = scores2[nms_keep2]
        else:
            boxes2 = boxes1
            labels2 = labels1
            scores2 = scores1
        
        pred2 = {
            'boxes': boxes2.cpu(),
            'labels': labels2.cpu(),
            'scores': scores2.cpu()
        }
        
        results.append((pred1, pred2))
    
    return results[0]


def check_predictions_different(pred1, pred2, threshold=0.1):
    """
    检查两个预测结果是否不同
    
    判断标准：
    1. 检测框数量不同
    2. 或者类别分布不同
    3. 或者置信度差异较大
    """
    boxes1 = pred1['boxes']
    boxes2 = pred2['boxes']
    labels1 = pred1['labels']
    labels2 = pred2['labels']
    scores1 = pred1['scores']
    scores2 = pred2['scores']
    
    # 检查数量
    if len(boxes1) != len(boxes2):
        return True, f"数量不同: {len(boxes1)} vs {len(boxes2)}"
    
    # 检查类别分布
    if len(labels1) > 0 and len(labels2) > 0:
        unique1 = set(labels1.tolist())
        unique2 = set(labels2.tolist())
        if unique1 != unique2:
            return True, f"类别不同: {unique1} vs {unique2}"
    
    # 检查平均置信度
    if len(scores1) > 0 and len(scores2) > 0:
        avg_score1 = scores1.mean().item()
        avg_score2 = scores2.mean().item()
        if abs(avg_score1 - avg_score2) > threshold:
            return True, f"平均置信度差异: {avg_score1:.3f} vs {avg_score2:.3f}"
    
    return False, "相同"


def draw_predictions_on_image(image, predictions, scale, pad_info, conf_thresh, input_size, title=""):
    """在图片上绘制预测结果"""
    vis_image = image.copy()
    h, w = image.shape[:2]
    pad_left, pad_top = pad_info
    
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_thresh:
            continue
        
        cx, cy, bw, bh = box
        x1_padded = (cx - bw / 2) * input_size
        y1_padded = (cy - bh / 2) * input_size
        x2_padded = (cx + bw / 2) * input_size
        y2_padded = (cy + bh / 2) * input_size
        
        x1_scaled = x1_padded - pad_left
        y1_scaled = y1_padded - pad_top
        x2_scaled = x2_padded - pad_left
        y2_scaled = y2_padded - pad_top
        
        x1 = int(x1_scaled / scale)
        y1 = int(y1_scaled / scale)
        x2 = int(x2_scaled / scale)
        y2 = int(y2_scaled / scale)
        
        label_id = int(label.item())
        class_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else f'class_{label_id}'
        color = COLORS[label_id % len(COLORS)]
        
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        label_text = f'{class_name} {score:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(vis_image, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 添加标题
    if title:
        cv2.putText(vis_image, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    return vis_image


def create_comparison_image(orig_image, pred1, pred2, scale, pad_info, conf_thresh, input_size, diff_reason):
    """创建对比图（左右并排）"""
    # 绘制两个版本
    img1 = draw_predictions_on_image(
        orig_image, pred1, scale, pad_info, conf_thresh, input_size,
        title=f"Only Conf (n={len(pred1['boxes'])})"
    )
    img2 = draw_predictions_on_image(
        orig_image, pred2, scale, pad_info, conf_thresh, input_size,
        title=f"Conf+Semantic (n={len(pred2['boxes'])})"
    )
    
    # 横向拼接
    h, w = img1.shape[:2]
    combined = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    combined[:, :w] = img1
    combined[:, w+20:] = img2
    
    # 添加中间分隔线
    combined[:, w:w+20] = 255
    
    # 添加差异说明
    reason_text = f"Difference: {diff_reason}"
    cv2.putText(combined, reason_text, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='批量测试并对比语义过滤前后差异')
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth',
                       help='检查点文件路径')
    parser.add_argument('--image-dir', type=str,
                       default='/data/usb/coco/train2017/',
                       help='图片目录')
    parser.add_argument('--config', type=str,
                       default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str,
                       default='visualizations/semantic_compare/',
                       help='输出目录')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                       help='置信度阈值')
    parser.add_argument('--proto-percentile', type=int, default=50,
                       help='语义阈值百分位数')
    parser.add_argument('--nms-iou', type=float, default=0.5,
                       help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备')
    parser.add_argument('--max-images', type=int, default=1000,
                       help='最大测试图片数量（0表示全部）')
    parser.add_argument('--prototype-path', type=str,
                       default='adaptive_prototypes.pth',
                       help='语义原型文件路径')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("🔍 批量测试 - 对比语义过滤前后差异")
    print(f"{'='*80}")
    print(f"图片目录: {args.image_dir}")
    print(f"输出目录: {args.output}")
    print(f"置信度阈值: {args.conf_thresh}")
    print(f"语义阈值: P{args.proto_percentile}")
    print(f"NMS IoU: {args.nms_iou}")
    print(f"{'='*80}\n")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("📦 加载模型...")
    model = load_model(args.checkpoint, args.config, device)
    print("✅ 模型加载完成\n")
    
    # 加载语义原型
    print("📦 加载语义原型...")
    from adaptive_prototypes import AdaptivePrototypes
    adaptive_proto = AdaptivePrototypes.load(args.prototype_path, device=device)
    print("✅ 语义原型加载完成\n")
    
    # 获取所有图片
    image_dir = Path(args.image_dir)
    image_files = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    
    if args.max_images > 0:
        image_files = image_files[:args.max_images]
    
    print(f"📂 找到 {len(image_files)} 张图片\n")
    
    # 统计信息
    total_processed = 0
    total_different = 0
    
    # 遍历图片
    print("🚀 开始批量测试...\n")
    
    for img_path in tqdm(image_files, desc="处理图片"):
        # 预处理
        image_tensor, orig_image, scale, pad_info = preprocess_image(img_path)
        
        if image_tensor is None:
            continue
        
        total_processed += 1
        
        try:
            # 执行推理（两种策略）
            pred1, pred2 = inference_with_two_strategies(
                model, image_tensor, device,
                args.conf_thresh, adaptive_proto,
                args.proto_percentile, args.nms_iou
            )
            
            # 检查是否不同
            is_different, diff_reason = check_predictions_different(pred1, pred2)
            
            if is_different:
                total_different += 1
                
                # 创建对比图
                comparison = create_comparison_image(
                    orig_image, pred1, pred2,
                    scale, pad_info, args.conf_thresh, 640, diff_reason
                )
                
                # 保存
                output_path = output_dir / f"{img_path.stem}_compare.jpg"
                cv2.imwrite(str(output_path), comparison)
                
                # 输出信息
                tqdm.write(f"✅ [{total_different}] {img_path.name}: {diff_reason}")
        
        except Exception as e:
            tqdm.write(f"❌ 处理失败 {img_path.name}: {e}")
            continue
    
    # 输出统计
    print(f"\n{'='*80}")
    print("📊 统计结果")
    print(f"{'='*80}")
    print(f"总处理图片: {total_processed}")
    print(f"发现差异: {total_different} ({total_different/total_processed*100:.1f}%)")
    print(f"结果相同: {total_processed - total_different} ({(total_processed-total_different)/total_processed*100:.1f}%)")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

