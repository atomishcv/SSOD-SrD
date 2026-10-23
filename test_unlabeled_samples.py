"""
从未标注数据中随机选取样本进行模型测试
要求选取的影像必须有检测结果，并保存可视化结果

使用方法:
    python test_unlabeled_samples.py -c configs/deimv2_ssod/deimv2_dinov3_x_coco.yml \
                                     --checkpoint outputs/deimv2_dinov3_x_dji_ssod/checkpoint_best_ap0.5842_epoch042.pth \
                                     --num-samples 30 \
                                     --conf-thresh 0.3
"""

import os
import sys
import argparse
import json
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from engine.core import YAMLConfig
import torchvision.transforms as T


def load_model_from_checkpoint(checkpoint_path: str, cfg, device: torch.device):
    """从检查点加载模型"""
    print(f"📂 加载检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = cfg.model.to(device)
    
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']['module']
        print("✅ 使用 EMA 模型权重")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("✅ 使用普通模型权重")
    else:
        state_dict = checkpoint
        print("✅ 直接加载 state_dict")
    
    # 处理 DDP 包装
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def preprocess_image(image_path: str, size: int = 800):
    """预处理图像"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # 调整大小
    image_resized = image.resize((size, size), Image.BILINEAR)
    
    # 转换为张量并归一化
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_resized)
    
    return image, image_tensor, original_size


def detect_single_image(model, postprocessor, image_tensor, original_size, device, conf_thresh=0.3):
    """对单张图像进行检测"""
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # 后处理
    orig_target_sizes = torch.tensor([[original_size[1], original_size[0]]], device=device)
    results = postprocessor(outputs, orig_target_sizes)
    
    # 过滤低置信度结果
    result = results[0]
    scores = result['scores']
    keep = scores >= conf_thresh
    
    boxes = result['boxes'][keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = result['labels'][keep].cpu().numpy()
    
    return boxes, scores, labels


def draw_detections(image: Image.Image, boxes, scores, labels, class_names=['target']):
    """在图像上绘制检测结果"""
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # 颜色
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        color = colors[label % len(colors)]
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签和分数
        class_name = class_names[label] if label < len(class_names) else f'class_{label}'
        text = f'{class_name}: {score:.2f}'
        
        # 计算文本背景
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        draw.text((x1, y1), text, fill='white', font=font)
    
    return image


def main(args):
    print("=" * 80)
    print("🔍 未标注数据样本检测测试")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 加载配置
    print(f"\n📋 加载配置: {args.config}")
    cfg = YAMLConfig(args.config, resume=args.checkpoint)
    
    # 加载模型
    model = load_model_from_checkpoint(args.checkpoint, cfg, device)
    postprocessor = cfg.postprocessor
    
    # 读取未标注数据列表
    unlabeled_json = '/data/DEIMv2/DJI_dataset/annotations/semi_supervised/train_unlabeled.json'
    print(f"\n📁 读取未标注数据: {unlabeled_json}")
    
    with open(unlabeled_json, 'r') as f:
        unlabeled_data = json.load(f)
    
    images_info = unlabeled_data['images']
    print(f"   - 总图像数量: {len(images_info)}")
    
    # 随机打乱图像列表
    random.shuffle(images_info)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 输出目录: {output_dir}")
    
    # 图像根目录
    image_root = '/data/DEIMv2/DJI_dataset/images'
    
    # 开始检测
    print(f"\n🚀 开始检测 (目标: {args.num_samples} 张有检测结果的图像)")
    print(f"   - 置信度阈值: {args.conf_thresh}")
    print("-" * 80)
    
    selected_images = []
    detection_results = []
    
    # 遍历图像直到找到足够的有检测结果的样本
    pbar = tqdm(images_info, desc="检测进度")
    
    for img_info in pbar:
        if len(selected_images) >= args.num_samples:
            break
        
        # 获取图像路径
        file_name = img_info['file_name']
        image_path = os.path.join(image_root, file_name)
        
        if not os.path.exists(image_path):
            continue
        
        try:
            # 预处理
            original_image, image_tensor, original_size = preprocess_image(image_path, size=800)
            
            # 检测
            boxes, scores, labels = detect_single_image(
                model, postprocessor, image_tensor, original_size, device, args.conf_thresh
            )
            
            # 只保存有检测结果的图像
            if len(boxes) > 0:
                sample_idx = len(selected_images) + 1
                stem_name = Path(file_name).stem
                
                # 保存原始图像
                original_save_name = f"sample_{sample_idx:03d}_{stem_name}_original.jpg"
                original_save_path = output_dir / original_save_name
                original_image.save(original_save_path, quality=95)
                
                # 绘制检测结果
                result_image = draw_detections(original_image.copy(), boxes, scores, labels)
                
                # 保存预测结果图像
                save_name = f"sample_{sample_idx:03d}_{stem_name}_det{len(boxes)}.jpg"
                save_path = output_dir / save_name
                result_image.save(save_path, quality=95)
                
                selected_images.append({
                    'file_name': file_name,
                    'image_id': img_info.get('id', len(selected_images)),
                    'num_detections': len(boxes),
                    'original_path': str(original_save_path),
                    'detection_path': str(save_path)
                })
                
                detection_results.append({
                    'file_name': file_name,
                    'image_id': img_info.get('id', len(selected_images)),
                    'detections': [
                        {
                            'box': box.tolist(),
                            'score': float(score),
                            'label': int(label)
                        }
                        for box, score, label in zip(boxes, scores, labels)
                    ]
                })
                
                pbar.set_postfix({
                    '已选取': len(selected_images),
                    '检测数': len(boxes)
                })
        
        except Exception as e:
            continue
    
    # 保存检测结果 JSON
    results_json_path = output_dir / 'detection_results.json'
    with open(results_json_path, 'w') as f:
        json.dump({
            'config': {
                'checkpoint': args.checkpoint,
                'conf_thresh': args.conf_thresh,
                'num_samples': len(selected_images)
            },
            'results': detection_results
        }, f, indent=2)
    
    # 打印统计
    print("\n" + "=" * 80)
    print("📊 检测结果统计")
    print("=" * 80)
    print(f"   - 成功选取图像: {len(selected_images)} 张")
    print(f"   - 总检测框数: {sum(r['num_detections'] for r in selected_images)}")
    print(f"   - 平均每张检测数: {sum(r['num_detections'] for r in selected_images) / len(selected_images):.2f}")
    print(f"\n📁 结果保存位置:")
    print(f"   - 原始图像: {output_dir}/*_original.jpg")
    print(f"   - 检测结果图像: {output_dir}/*_det*.jpg")
    print(f"   - 检测结果 JSON: {results_json_path}")
    
    # 打印每张图像的检测数量
    print(f"\n📋 各图像检测数量:")
    for i, img in enumerate(selected_images):
        print(f"   {i+1:2d}. {img['file_name']}: {img['num_detections']} 个检测")
    
    print("\n" + "=" * 80)
    print("✅ 检测完成!")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='未标注数据样本检测测试')
    parser.add_argument('-c', '--config', type=str, default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='outputs/deimv2_dinov3_x_dji_ssod/checkpoint_best_ap0.5842_epoch042.pth',
                        help='模型检查点路径')
    parser.add_argument('--num-samples', type=int, default=30,
                        help='要选取的样本数量 (默认: 30)')
    parser.add_argument('--conf-thresh', type=float, default=0.3,
                        help='置信度阈值 (默认: 0.3)')
    parser.add_argument('--output-dir', type=str, default='visualizations/unlabeled_samples',
                        help='输出目录 (默认: visualizations/unlabeled_samples)')
    
    args = parser.parse_args()
    
    main(args)

