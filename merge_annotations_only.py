"""
只合并标注文件，不复制图像
假设图像文件已经在 DJI_dataset/images/ 中
"""
import json
from pathlib import Path
import random

def fix_category_id(data):
    """修复category_id: 0→1"""
    if 'categories' in data:
        for cat in data['categories']:
            if cat['id'] == 0:
                cat['id'] = 1
    
    if 'annotations' in data:
        for ann in data['annotations']:
            if ann['category_id'] == 0:
                ann['category_id'] = 1
    
    return data

def merge_annotations():
    """只合并标注文件"""
    print("="*70)
    print("合并所有标注文件（跳过图像复制）")
    print("="*70 + "\n")
    
    base_path = Path('DJI_dataset')
    dst_ann_dir = base_path / 'annotations'
    dst_semi_dir = dst_ann_dir / 'semi_supervised'
    
    dst_ann_dir.mkdir(exist_ok=True)
    dst_semi_dir.mkdir(exist_ok=True)
    
    subsets = [
        {'name': 'coco_dataset', 'prefix': 'coco_dataset_'},
        {'name': 'coco_dataset_1', 'prefix': 'coco_dataset_1_'},
        {'name': 'coco_dataset_2', 'prefix': 'coco_dataset_2_'}
    ]
    
    # ========== 第一步：合并有标注数据 ==========
    print("步骤1: 合并有标注数据 (train_full.json)\n")
    
    all_labeled_images = []
    all_labeled_annotations = []
    image_id_counter = 1
    ann_id_counter = 1
    
    for subset in subsets:
        subset_path = base_path / subset['name']
        ann_file = subset_path / 'annotations' / 'train_full.json'
        
        if not ann_file.exists():
            print(f"  ⚠️  {subset['name']}: train_full.json不存在，跳过")
            continue
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        data = fix_category_id(data)
        
        print(f"  {subset['name']}:")
        print(f"    - 图像: {len(data['images'])}")
        print(f"    - 标注: {len(data['annotations'])}")
        
        # 创建ID映射
        old_to_new_id = {}
        for img in data['images']:
            old_id = img['id']
            new_id = image_id_counter
            old_to_new_id[old_id] = new_id
            
            new_img = {
                'id': new_id,
                'file_name': f"{subset['prefix']}{img['file_name']}",
                'width': img['width'],
                'height': img['height']
            }
            all_labeled_images.append(new_img)
            image_id_counter += 1
        
        for ann in data['annotations']:
            if ann['image_id'] not in old_to_new_id:
                continue
            
            new_ann = {
                'id': ann_id_counter,
                'image_id': old_to_new_id[ann['image_id']],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann.get('iscrowd', 0)
            }
            all_labeled_annotations.append(new_ann)
            ann_id_counter += 1
        
        print(f"    ✓ 已添加\n")
    
    print(f"  有标注数据合计: {len(all_labeled_images)} 张, {len(all_labeled_annotations)} 个标注\n")
    
    # ========== 第二步：合并无标注数据信息 ==========
    print("步骤2: 合并无标注图像信息 (unlabeled_images.json)\n")
    
    all_unlabeled_images = []
    unlabeled_id_counter = image_id_counter
    
    for subset in subsets:
        subset_path = base_path / subset['name']
        unlabeled_file = subset_path / 'annotations' / 'unlabeled_images.json'
        
        if not unlabeled_file.exists():
            print(f"  ⚠️  {subset['name']}: unlabeled_images.json不存在，跳过")
            continue
        
        with open(unlabeled_file, 'r') as f:
            data = json.load(f)
        
        data = fix_category_id(data)
        
        print(f"  {subset['name']}: {len(data['images'])} 张无标注图像")
        
        for img in data['images']:
            new_img = {
                'id': unlabeled_id_counter,
                'file_name': f"{subset['prefix']}{img['file_name']}",
                'width': img['width'],
                'height': img['height']
            }
            all_unlabeled_images.append(new_img)
            unlabeled_id_counter += 1
    
    print(f"\n  无标注图像合计: {len(all_unlabeled_images)} 张\n")
    
    # ========== 第三步：划分训练/验证集 ==========
    print("步骤3: 划分训练集和验证集\n")
    
    random.seed(42)
    labeled_indices = list(range(len(all_labeled_images)))
    random.shuffle(labeled_indices)
    
    split_idx = int(len(labeled_indices) * 0.85)
    train_indices = set(labeled_indices[:split_idx])
    val_indices = set(labeled_indices[split_idx:])
    
    # 训练集
    train_images = [all_labeled_images[i] for i in train_indices]
    train_img_ids = set(img['id'] for img in train_images)
    train_annotations = [ann for ann in all_labeled_annotations if ann['image_id'] in train_img_ids]
    
    # 验证集
    val_images = [all_labeled_images[i] for i in val_indices]
    val_img_ids = set(img['id'] for img in val_images)
    val_annotations = [ann for ann in all_labeled_annotations if ann['image_id'] in val_img_ids]
    
    print(f"  训练集: {len(train_images)} 张, {len(train_annotations)} 个标注")
    print(f"  验证集: {len(val_images)} 张, {len(val_annotations)} 个标注")
    print()
    
    # ========== 第四步：保存所有标注文件 ==========
    print("步骤4: 保存标注文件\n")
    
    categories = [{'id': 1, 'name': 'target', 'supercategory': 'none'}]
    
    # 1. train_full_merged.json（所有有标注数据）
    train_full_data = {
        'info': {'description': 'DJI Complete Merged - All Labeled', 'version': '2.0'},
        'licenses': [],
        'categories': categories,
        'images': all_labeled_images,
        'annotations': all_labeled_annotations
    }
    
    with open(dst_ann_dir / 'train_full_merged.json', 'w') as f:
        json.dump(train_full_data, f, indent=2)
    print(f"  ✅ train_full_merged.json (所有有标注数据)")
    
    # 2. train_all_merged.json（训练集）
    train_data = {
        'info': {'description': 'DJI Complete Merged - Train Split', 'version': '2.0'},
        'licenses': [],
        'categories': categories,
        'images': train_images,
        'annotations': train_annotations
    }
    
    with open(dst_ann_dir / 'train_all_merged.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  ✅ train_all_merged.json (训练集)")
    
    # 3. val_all_merged.json（验证集）
    val_data = {
        'info': {'description': 'DJI Complete Merged - Val Split', 'version': '2.0'},
        'licenses': [],
        'categories': categories,
        'images': val_images,
        'annotations': val_annotations
    }
    
    with open(dst_ann_dir / 'val_all_merged.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  ✅ val_all_merged.json (验证集)")
    
    # 4. unlabeled_images_merged.json（所有无标注图像）
    unlabeled_data = {
        'info': {'description': 'DJI Complete Merged - Unlabeled Images', 'version': '2.0'},
        'licenses': [],
        'categories': categories,
        'images': all_unlabeled_images
    }
    
    with open(dst_ann_dir / 'unlabeled_images_merged.json', 'w') as f:
        json.dump(unlabeled_data, f, indent=2)
    print(f"  ✅ unlabeled_images_merged.json (无标注图像)")
    
    # 5. semi_supervised/train_labeled_merged.json
    with open(dst_semi_dir / 'train_labeled_merged.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  ✅ semi_supervised/train_labeled_merged.json")
    
    # 6. semi_supervised/train_unlabeled_merged.json
    with open(dst_semi_dir / 'train_unlabeled_merged.json', 'w') as f:
        json.dump(unlabeled_data, f, indent=2)
    print(f"  ✅ semi_supervised/train_unlabeled_merged.json")
    
    print()
    print("="*70)
    print("✅ 标注文件合并完成！")
    print("="*70)
    print(f"\n📊 最终数据统计:")
    print(f"  有标注图像: {len(all_labeled_images)} 张")
    print(f"    - 训练集: {len(train_images)} 张, {len(train_annotations)} 个标注")
    print(f"    - 验证集: {len(val_images)} 张, {len(val_annotations)} 个标注")
    print(f"  无标注图像: {len(all_unlabeled_images)} 张")
    print()
    print(f"📁 文件位置:")
    print(f"  DJI_dataset/annotations/")
    print(f"    ├── train_full_merged.json (完整有标注)")
    print(f"    ├── train_all_merged.json (训练集)")
    print(f"    ├── val_all_merged.json (验证集)")
    print(f"    ├── unlabeled_images_merged.json (无标注)")
    print(f"    └── semi_supervised/")
    print(f"        ├── train_labeled_merged.json")
    print(f"        └── train_unlabeled_merged.json")
    print()
    print(f"🚀 可以开始训练:")
    print(f"  全监督: python train.py")
    print(f"  半监督: python ssod_train.py")
    print()

if __name__ == '__main__':
    merge_annotations()



