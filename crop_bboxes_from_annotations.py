#!/usr/bin/env python3
"""
根据标注文件裁剪目标框并保存
"""

import json
import os
from pathlib import Path
from PIL import Image
import argparse

def crop_bboxes_from_annotations(
    ann_file,
    img_folder,
    output_dir,
    min_size=10,
    save_with_category=True,
    max_crops_per_image=None
):
    """
    根据标注文件裁剪目标框
    
    Args:
        ann_file: 标注JSON文件路径
        img_folder: 图片文件夹路径
        output_dir: 输出文件夹路径
        min_size: 最小裁剪尺寸（过滤太小的框）
        save_with_category: 是否按类别分类保存
        max_crops_per_image: 每张图片最多裁剪数量（None表示全部）
    """
    # 读取标注文件
    print(f"Loading annotations from: {ann_file}")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    
    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Categories: {categories}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_with_category:
        for cat_id, cat_name in categories.items():
            (output_dir / cat_name).mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'total_crops': 0,
        'skipped_small': 0,
        'skipped_missing_image': 0,
        'by_category': {cat_id: 0 for cat_id in categories.keys()}
    }
    
    # 按图片分组标注
    image_id_to_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)
    
    # 处理每张图片
    for image_id, img_info in images.items():
        if image_id not in image_id_to_annotations:
            continue
        
        img_file = Path(img_folder) / img_info['file_name']
        if not img_file.exists():
            print(f"Warning: Image not found: {img_file}")
            stats['skipped_missing_image'] += len(image_id_to_annotations[image_id])
            continue
        
        # 读取图片
        try:
            img = Image.open(img_file)
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            continue
        
        # 获取该图片的标注
        img_annotations = image_id_to_annotations[image_id]
        if max_crops_per_image:
            img_annotations = img_annotations[:max_crops_per_image]
        
        # 裁剪每个目标框
        for ann_idx, ann in enumerate(img_annotations):
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # 过滤太小的框
            if w < min_size or h < min_size:
                stats['skipped_small'] += 1
                continue
            
            # 转换为整数并确保在图片范围内
            x = int(max(0, x))
            y = int(max(0, y))
            w = int(min(w, img.width - x))
            h = int(min(h, img.height - y))
            
            if w <= 0 or h <= 0:
                continue
            
            # 裁剪
            crop = img.crop((x, y, x + w, y + h))
            
            # 生成保存文件名
            img_name = Path(img_info['file_name']).stem
            cat_id = ann.get('category_id', 0)
            cat_name = categories.get(cat_id, f'category_{cat_id}')
            
            if save_with_category:
                save_dir = output_dir / cat_name
                save_name = f"{img_name}_img{image_id}_ann{ann['id']}_cat{cat_id}.jpg"
            else:
                save_dir = output_dir
                save_name = f"{img_name}_img{image_id}_ann{ann['id']}_cat{cat_name}.jpg"
            
            save_path = save_dir / save_name
            
            # 保存裁剪图片
            crop.save(save_path, quality=95)
            stats['total_crops'] += 1
            stats['by_category'][cat_id] += 1
        
        if (list(images.keys()).index(image_id) + 1) % 100 == 0:
            print(f"Processed {list(images.keys()).index(image_id) + 1}/{len(images)} images...")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("Crop Statistics:")
    print("="*50)
    print(f"Total crops saved: {stats['total_crops']}")
    print(f"Skipped (too small): {stats['skipped_small']}")
    print(f"Skipped (missing image): {stats['skipped_missing_image']}")
    print("\nBy category:")
    for cat_id, count in stats['by_category'].items():
        cat_name = categories.get(cat_id, f'category_{cat_id}')
        print(f"  {cat_name} (id={cat_id}): {count} crops")
    print(f"\nOutput directory: {output_dir}")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop bounding boxes from annotations')
    parser.add_argument('--ann-file', type=str, 
                       default='DJI_dataset/annotations/train_full.json',
                       help='Annotation JSON file path')
    parser.add_argument('--img-folder', type=str,
                       default='DJI_dataset/images',
                       help='Image folder path')
    parser.add_argument('--output-dir', type=str,
                       default='DJI_dataset/cropped_bboxes',
                       help='Output directory for cropped images')
    parser.add_argument('--min-size', type=int, default=10,
                       help='Minimum crop size (width or height)')
    parser.add_argument('--no-category-dirs', action='store_true',
                       help='Do not organize by category directories')
    parser.add_argument('--max-crops-per-image', type=int, default=None,
                       help='Maximum number of crops per image (None for all)')
    
    args = parser.parse_args()
    
    crop_bboxes_from_annotations(
        ann_file=args.ann_file,
        img_folder=args.img_folder,
        output_dir=args.output_dir,
        min_size=args.min_size,
        save_with_category=not args.no_category_dirs,
        max_crops_per_image=args.max_crops_per_image
    )

