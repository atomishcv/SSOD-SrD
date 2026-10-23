#!/usr/bin/env python3
"""
可视化 COCO 格式的标注文件
在图片上绘制边界框和类别标签
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np


def load_coco_annotations(ann_file):
    """加载 COCO 格式的标注文件"""
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    return coco_data


def get_image_annotations(coco_data, image_id):
    """获取指定图片的所有标注"""
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            annotations.append(ann)
    return annotations


def get_category_name(coco_data, category_id):
    """根据类别 ID 获取类别名称"""
    for cat in coco_data['categories']:
        if cat['id'] == category_id:
            return cat['name']
    return f'class_{category_id}'


def visualize_image(image_path, annotations, coco_data, output_path=None, show=True):
    """
    可视化单张图片及其标注
    
    Args:
        image_path: 图片路径
        annotations: 该图片的标注列表
        coco_data: COCO 数据字典
        output_path: 输出路径（可选）
        show: 是否显示图片
    """
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_array)
    ax.axis('off')
    
    # 颜色映射（为不同类别分配不同颜色）
    colors = plt.cm.Set3(np.linspace(0, 1, len(coco_data['categories'])))
    
    # 绘制每个标注
    for ann in annotations:
        # COCO bbox 格式: [x, y, width, height]
        x, y, w, h = ann['bbox']
        
        # 获取类别信息
        category_id = ann['category_id']
        category_name = get_category_name(coco_data, category_id)
        
        # 选择颜色
        color = colors[category_id - 1] if category_id <= len(colors) else colors[0]
        color_rgb = tuple(c[:3] for c in [color])[0]
        
        # 绘制边界框
        rect = Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color_rgb,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签文本
        label_text = f"{category_name}"
        if 'size_category' in ann:
            label_text += f" ({ann['size_category']})"
        
        ax.text(
            x, y - 5,
            label_text,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color_rgb, alpha=0.7),
            fontsize=10,
            color='black',
            weight='bold'
        )
    
    # 设置标题
    image_name = os.path.basename(image_path)
    num_anns = len(annotations)
    ax.set_title(f'{image_name}\n标注数量: {num_anns}', fontsize=12, pad=10)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {output_path}")
    
    # 显示图片
    if show:
        plt.show()
    else:
        plt.close()


def visualize_all_images(images_dir, ann_file, output_dir=None, max_images=None, show=False):
    """
    可视化所有图片的标注
    
    Args:
        images_dir: 图片目录
        ann_file: 标注文件路径
        output_dir: 输出目录（可选）
        max_images: 最大可视化图片数量（可选）
        show: 是否显示图片
    """
    # 加载标注数据
    coco_data = load_coco_annotations(ann_file)
    
    # 创建输出目录
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建图片 ID 到文件名的映射
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # 统计信息
    total_images = len(coco_data['images'])
    images_to_visualize = list(coco_data['images'])
    
    if max_images:
        images_to_visualize = images_to_visualize[:max_images]
    
    print(f"📊 总共 {total_images} 张图片，将可视化 {len(images_to_visualize)} 张")
    print(f"📁 图片目录: {images_dir}")
    print(f"📄 标注文件: {ann_file}")
    if output_dir:
        print(f"💾 输出目录: {output_dir}")
    print()
    
    # 可视化每张图片
    for idx, image_info in enumerate(images_to_visualize, 1):
        image_id = image_info['id']
        file_name = image_info['file_name']
        image_path = Path(images_dir) / file_name
        
        if not image_path.exists():
            print(f"⚠️  跳过 {file_name}: 文件不存在")
            continue
        
        # 获取该图片的标注
        annotations = get_image_annotations(coco_data, image_id)
        
        # 确定输出路径
        output_path = None
        if output_dir:
            output_name = f"{Path(file_name).stem}_visualized.png"
            output_path = output_dir / output_name
        
        print(f"[{idx}/{len(images_to_visualize)}] 处理: {file_name} ({len(annotations)} 个标注)")
        
        # 可视化
        visualize_image(
            image_path,
            annotations,
            coco_data,
            output_path=output_path,
            show=show
        )
    
    print(f"\n✅ 可视化完成！")


def main():
    parser = argparse.ArgumentParser(description='可视化 COCO 格式的标注文件')
    parser.add_argument(
        '--images-dir',
        type=str,
        default='images',
        help='图片目录路径'
    )
    parser.add_argument(
        '--ann-file',
        type=str,
        default='annotations/val_labeled.json',
        help='COCO 格式标注文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations/gt_label',
        help='输出目录（保存可视化结果）'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='最大可视化图片数量（默认：全部）'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='显示图片（默认：仅保存）'
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 处理相对路径
    images_dir = script_dir / args.images_dir if not Path(args.images_dir).is_absolute() else Path(args.images_dir)
    ann_file = script_dir / args.ann_file if not Path(args.ann_file).is_absolute() else Path(args.ann_file)
    output_dir = script_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    
    # 检查文件是否存在
    if not images_dir.exists():
        print(f"❌ 错误: 图片目录不存在: {images_dir}")
        return
    
    if not ann_file.exists():
        print(f"❌ 错误: 标注文件不存在: {ann_file}")
        return
    
    # 可视化
    visualize_all_images(
        images_dir,
        ann_file,
        output_dir=output_dir,
        max_images=args.max_images,
        show=args.show
    )


if __name__ == "__main__":
    main()
