#!/usr/bin/env python3
"""
将 labeled_folders 中的 LabelMe 格式标注转换为 COCO 格式
输出结果保存到 labeled_folders 目录下
"""

import os
import json
from pathlib import Path
from PIL import Image
from datetime import datetime


def convert_labelme_to_coco(input_base_dir: Path, output_dir: Path):
    """
    将 LabelMe 格式的标注文件转换为 COCO 格式
    
    Args:
        input_base_dir: 包含子目录的输入目录
        output_dir: 输出目录
    """
    
    # 创建输出目录结构
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO 格式的基本结构
    coco_data = {
        "info": {
            "description": "Labeled dataset converted from LabelMe format",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Auto-generated",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "plant",
                "supercategory": "none"
            }
        ]
    }
    
    image_id = 0
    annotation_id = 0
    
    # 标签映射：LabelMe 中的 target -> COCO 中的 plant (id=1)
    label_mapping = {
        "target": 1,
        "plant": 1
    }
    
    # 遍历所有子目录
    subdirs = sorted([d for d in input_base_dir.iterdir() if d.is_dir() and d.name.startswith("DJI_")])
    
    print(f"📂 找到 {len(subdirs)} 个子目录")
    
    for subdir in subdirs:
        # 查找所有 JSON 文件
        json_files = sorted(subdir.glob("*.json"))
        
        for json_file in json_files:
            # 对应的图片文件
            image_file = json_file.with_suffix(".jpg")
            if not image_file.exists():
                print(f"⚠️  跳过 {json_file.name}: 找不到对应的图片")
                continue
            
            # 读取 LabelMe JSON
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    labelme_data = json.load(f)
            except Exception as e:
                print(f"⚠️  跳过 {json_file.name}: 读取失败 - {e}")
                continue
            
            # 生成唯一的文件名
            unique_filename = f"{subdir.name}_{image_file.name}"
            
            # 获取图片尺寸
            try:
                with Image.open(image_file) as img:
                    width, height = img.size
            except Exception as e:
                print(f"⚠️  跳过 {json_file.name}: 无法读取图片尺寸 - {e}")
                continue
            
            # 复制图片到输出目录
            import shutil
            dest_image_path = images_dir / unique_filename
            shutil.copy2(image_file, dest_image_path)
            
            image_id += 1
            
            # 添加图片信息
            coco_data["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": unique_filename
            })
            
            # 处理标注
            shapes = labelme_data.get("shapes", [])
            for shape in shapes:
                label = shape.get("label", "").lower()
                points = shape.get("points", [])
                shape_type = shape.get("shape_type", "")
                
                # 获取类别 ID
                category_id = label_mapping.get(label)
                if category_id is None:
                    print(f"⚠️  未知标签 '{label}' in {json_file.name}")
                    continue
                
                # 计算边界框
                if shape_type == "rectangle" and len(points) >= 2:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)
                elif shape_type == "polygon" and len(points) >= 3:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)
                else:
                    print(f"⚠️  不支持的形状类型 '{shape_type}' in {json_file.name}")
                    continue
                
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                area = bbox_width * bbox_height
                
                # 跳过无效的边界框
                if bbox_width <= 0 or bbox_height <= 0:
                    continue
                
                annotation_id += 1
                
                # 添加标注信息 (COCO bbox 格式: [x, y, width, height])
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": area,
                    "iscrowd": 0,
                    "size_category": "small" if area < 32*32 else ("medium" if area < 96*96 else "large")
                })
    
    # 保存 COCO 格式的标注文件
    output_ann_file = annotations_dir / "val_labeled.json"
    with open(output_ann_file, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 转换完成!")
    print(f"   - 图片数量: {len(coco_data['images'])}")
    print(f"   - 标注数量: {len(coco_data['annotations'])}")
    print(f"   - 图片目录: {images_dir}")
    print(f"   - 标注文件: {output_ann_file}")
    
    return coco_data


def main():
    # 输入目录：labeled_folders 下的子目录
    script_dir = Path(__file__).parent
    input_base_dir = script_dir
    
    # 输出目录：labeled_folders 下
    output_dir = script_dir
    
    print(f"📁 输入目录: {input_base_dir}")
    print(f"📁 输出目录: {output_dir}")
    print()
    
    convert_labelme_to_coco(input_base_dir, output_dir)


if __name__ == "__main__":
    main()
