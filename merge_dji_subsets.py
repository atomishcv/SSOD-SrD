"""
将 DJI_dataset 下的三个子文件夹合并到主文件夹
- DJI_dataset/coco_dataset
- DJI_dataset/coco_dataset_1  
- DJI_dataset/coco_dataset_2

统一整合到:
- DJI_dataset/annotations/
- DJI_dataset/images/
"""
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def fix_category_id(data):
    """修复category_id: 0→1"""
    fixed = False
    if 'categories' in data:
        for cat in data['categories']:
            if cat['id'] == 0:
                cat['id'] = 1
                fixed = True
    
    if 'annotations' in data:
        for ann in data['annotations']:
            if ann['category_id'] == 0:
                ann['category_id'] = 1
                fixed = True
    
    return data, fixed

def merge_subsets(base_dir='DJI_dataset', val_ratio=0.15):
    """
    合并三个子文件夹到主文件夹
    
    Args:
        base_dir: DJI_dataset基础目录
        val_ratio: 验证集比例
    """
    print("="*70)
    print("合并DJI_dataset子文件夹")
    print("="*70 + "\n")
    
    base_path = Path(base_dir)
    
    # 三个子文件夹
    subsets = ['coco_dataset', 'coco_dataset_1', 'coco_dataset_2']
    
    # 收集所有数据
    all_images = []
    all_annotations = []
    image_id_counter = 1
    ann_id_counter = 1
    
    categories = [{'id': 1, 'name': 'target', 'supercategory': 'none'}]
    
    for subset_name in subsets:
        subset_path = base_path / subset_name
        ann_file = subset_path / 'annotations' / 'train_full.json'
        img_dir = subset_path / 'images'
        
        if not ann_file.exists():
            print(f"⚠️  {subset_name}: 标注文件不存在，跳过\n")
            continue
        
        print(f"处理 {subset_name}:")
        
        # 读取标注
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # 修复category_id
        data, fixed = fix_category_id(data)
        if fixed:
            print(f"  ✓ 已修复 category_id (0→1)")
        
        print(f"  - 图像数量: {len(data['images'])}")
        print(f"  - 标注数量: {len(data['annotations'])}")
        
        # 获取有标注的图像
        img_ids_with_ann = set(ann['image_id'] for ann in data['annotations'])
        print(f"  - 有标注图像: {len(img_ids_with_ann)}")
        
        # 创建ID映射
        old_to_new_img_id = {}
        
        # 处理图像
        for img in data['images']:
            if img['id'] not in img_ids_with_ann:
                continue  # 跳过没有标注的图像
            
            old_id = img['id']
            new_id = image_id_counter
            old_to_new_img_id[old_id] = new_id
            
            new_img = {
                'id': new_id,
                'file_name': f"{subset_name}_{img['file_name']}",  # 添加子集前缀
                'width': img['width'],
                'height': img['height']
            }
            all_images.append(new_img)
            image_id_counter += 1
        
        # 处理标注
        for ann in data['annotations']:
            if ann['image_id'] not in old_to_new_img_id:
                continue
            
            new_ann = {
                'id': ann_id_counter,
                'image_id': old_to_new_img_id[ann['image_id']],
                'category_id': ann['category_id'],  # 已经是1了
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann.get('iscrowd', 0)
            }
            all_annotations.append(new_ann)
            ann_id_counter += 1
        
        print(f"  ✓ 已添加 {len(old_to_new_img_id)} 张图像\n")
    
    print("="*70)
    print(f"合计: {len(all_images)} 张图像, {len(all_annotations)} 个标注")
    print("="*70 + "\n")
    
    # 划分训练集和验证集
    random.seed(42)
    indices = list(range(len(all_images)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - val_ratio))
    train_indices = set(indices[:split_idx])
    val_indices = set(indices[split_idx:])
    
    # 创建训练集
    train_images = [all_images[i] for i in train_indices]
    train_img_ids = set(img['id'] for img in train_images)
    train_annotations = [ann for ann in all_annotations if ann['image_id'] in train_img_ids]
    
    train_data = {
        'info': {'description': 'DJI Merged Dataset', 'version': '1.0', 'year': 2025},
        'licenses': [],
        'categories': categories,
        'images': train_images,
        'annotations': train_annotations
    }
    
    # 创建验证集
    val_images = [all_images[i] for i in val_indices]
    val_img_ids = set(img['id'] for img in val_images)
    val_annotations = [ann for ann in all_annotations if ann['image_id'] in val_img_ids]
    
    val_data = {
        'info': {'description': 'DJI Merged Dataset', 'version': '1.0', 'year': 2025},
        'licenses': [],
        'categories': categories,
        'images': val_images,
        'annotations': val_annotations
    }
    
    print(f"划分结果:")
    print(f"  训练集: {len(train_images)} 张, {len(train_annotations)} 个标注")
    print(f"  验证集: {len(val_images)} 张, {len(val_annotations)} 个标注")
    print()
    
    # 保存标注文件
    ann_dir = base_path / 'annotations'
    img_dir = base_path / 'images'
    
    train_ann_file = ann_dir / 'train_all_merged.json'
    val_ann_file = ann_dir / 'val_all_merged.json'
    
    with open(train_ann_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_ann_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✅ 标注文件已保存:")
    print(f"  - {train_ann_file}")
    print(f"  - {val_ann_file}")
    print()
    
    # 复制图像文件
    print("开始复制图像文件到 DJI_dataset/images...")
    print()
    
    total_copied = 0
    for subset_name in subsets:
        subset_path = base_path / subset_name
        ann_file = subset_path / 'annotations' / 'train_full.json'
        src_img_dir = subset_path / 'images'
        
        if not ann_file.exists() or not src_img_dir.exists():
            continue
        
        # 读取标注找出需要的图像
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        img_ids_with_ann = set(ann['image_id'] for ann in data['annotations'])
        images_to_copy = [img for img in data['images'] if img['id'] in img_ids_with_ann]
        
        print(f"复制 {subset_name} ({len(images_to_copy)} 个文件)...")
        
        copied = 0
        for img in tqdm(images_to_copy, desc=f"  {subset_name}"):
            src_file = src_img_dir / img['file_name']
            dst_file = img_dir / f"{subset_name}_{img['file_name']}"
            
            if src_file.exists() and not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                copied += 1
        
        print(f"  ✓ 已复制 {copied} 个文件\n")
        total_copied += copied
    
    print("="*70)
    print(f"✅ 整合完成！总共复制了 {total_copied} 个图像文件")
    print("="*70)
    print()
    print("最终数据集:")
    print(f"  - 位置: {base_dir}/")
    print(f"  - 图像: images/ ({total_copied} 个文件)")
    print(f"  - 训练标注: annotations/train_all_merged.json")
    print(f"  - 验证标注: annotations/val_all_merged.json")
    print()
    
    return train_ann_file, val_ann_file

if __name__ == '__main__':
    merge_subsets('DJI_dataset', val_ratio=0.15)

