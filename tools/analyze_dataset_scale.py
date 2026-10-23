"""
分析数据集中目标的尺度分布，用于确定单株/群落的划分阈值

使用方法:
    python tools/analyze_dataset_scale.py --ann_file DJI_dataset/annotations/train_all_merged.json

输出:
    - 目标面积分布直方图
    - 建议的阈值（基于聚类或百分位数）
"""

import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict


def analyze_dataset(ann_file: str, output_dir: str = None):
    """分析数据集中目标的尺度分布"""
    
    print(f"加载标注文件: {ann_file}")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # 收集所有目标的尺寸信息
    image_sizes = {img['id']: (img['width'], img['height']) for img in data['images']}
    
    areas_absolute = []  # 绝对面积（像素）
    areas_relative = []  # 相对面积（归一化）
    widths_relative = []
    heights_relative = []
    aspect_ratios = []
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_sizes:
            continue
        
        img_w, img_h = image_sizes[img_id]
        x, y, w, h = ann['bbox']  # COCO格式: x, y, w, h
        
        # 绝对面积
        area_abs = w * h
        areas_absolute.append(area_abs)
        
        # 相对面积（归一化到[0,1]）
        w_rel = w / img_w
        h_rel = h / img_h
        area_rel = w_rel * h_rel
        areas_relative.append(area_rel)
        widths_relative.append(w_rel)
        heights_relative.append(h_rel)
        
        # 宽高比
        if h > 0:
            aspect_ratios.append(w / h)
    
    areas_absolute = np.array(areas_absolute)
    areas_relative = np.array(areas_relative)
    widths_relative = np.array(widths_relative)
    heights_relative = np.array(heights_relative)
    aspect_ratios = np.array(aspect_ratios)
    
    print(f"\n{'='*60}")
    print(f"数据集统计")
    print(f"{'='*60}")
    print(f"图像数量: {len(data['images'])}")
    print(f"标注数量: {len(areas_relative)}")
    
    print(f"\n--- 相对面积统计 (归一化) ---")
    print(f"  最小值: {areas_relative.min():.6f}")
    print(f"  最大值: {areas_relative.max():.6f}")
    print(f"  均值: {areas_relative.mean():.6f}")
    print(f"  中位数: {np.median(areas_relative):.6f}")
    print(f"  标准差: {areas_relative.std():.6f}")
    
    # 计算百分位数
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n--- 百分位数 ---")
    for p in percentiles:
        val = np.percentile(areas_relative, p)
        print(f"  {p}%: {val:.6f}")
    
    # 使用K-Means聚类确定阈值
    print(f"\n--- K-Means聚类分析 (k=2) ---")
    areas_log = np.log(areas_relative + 1e-8).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(areas_log)
    
    cluster_0_mean = areas_relative[labels == 0].mean()
    cluster_1_mean = areas_relative[labels == 1].mean()
    
    if cluster_0_mean < cluster_1_mean:
        small_cluster, large_cluster = 0, 1
    else:
        small_cluster, large_cluster = 1, 0
    
    small_areas = areas_relative[labels == small_cluster]
    large_areas = areas_relative[labels == large_cluster]
    
    print(f"  小目标簇: {len(small_areas)} 个, 均值面积: {small_areas.mean():.6f}")
    print(f"  大目标簇: {len(large_areas)} 个, 均值面积: {large_areas.mean():.6f}")
    
    # 建议的阈值：两个簇之间的边界
    suggested_threshold = (small_areas.max() + large_areas.min()) / 2
    print(f"\n  建议阈值 (簇边界): {suggested_threshold:.6f}")
    
    # 另一种方法：使用百分位数
    threshold_75 = np.percentile(areas_relative, 75)
    threshold_90 = np.percentile(areas_relative, 90)
    print(f"  建议阈值 (75%分位): {threshold_75:.6f}")
    print(f"  建议阈值 (90%分位): {threshold_90:.6f}")
    
    # 绘制分布图
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 相对面积分布（对数尺度）
        ax = axes[0, 0]
        ax.hist(np.log10(areas_relative + 1e-8), bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.log10(suggested_threshold), color='r', linestyle='--', label=f'建议阈值: {suggested_threshold:.4f}')
        ax.set_xlabel('log10(相对面积)')
        ax.set_ylabel('数量')
        ax.set_title('目标相对面积分布 (对数尺度)')
        ax.legend()
        
        # 2. 相对面积分布（线性尺度，截断）
        ax = axes[0, 1]
        ax.hist(areas_relative[areas_relative < 0.1], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(suggested_threshold, color='r', linestyle='--', label=f'建议阈值: {suggested_threshold:.4f}')
        ax.set_xlabel('相对面积')
        ax.set_ylabel('数量')
        ax.set_title('目标相对面积分布 (面积<0.1)')
        ax.legend()
        
        # 3. 宽度 vs 高度散点图
        ax = axes[1, 0]
        scatter = ax.scatter(widths_relative, heights_relative, c=areas_relative, 
                            cmap='viridis', alpha=0.5, s=10)
        ax.set_xlabel('相对宽度')
        ax.set_ylabel('相对高度')
        ax.set_title('目标尺寸分布')
        plt.colorbar(scatter, ax=ax, label='相对面积')
        
        # 4. 聚类结果
        ax = axes[1, 1]
        ax.scatter(np.arange(len(small_areas)), np.sort(small_areas), 
                  alpha=0.5, s=5, label=f'小目标 (n={len(small_areas)})')
        ax.scatter(np.arange(len(large_areas)) + len(small_areas), np.sort(large_areas), 
                  alpha=0.5, s=5, label=f'大目标 (n={len(large_areas)})')
        ax.axhline(suggested_threshold, color='r', linestyle='--', label=f'建议阈值')
        ax.set_xlabel('目标索引 (排序后)')
        ax.set_ylabel('相对面积')
        ax.set_title('K-Means聚类结果')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        fig_path = output_dir / 'scale_distribution.png'
        plt.savefig(fig_path, dpi=150)
        print(f"\n分布图已保存: {fig_path}")
        plt.close()
    
    # 返回建议的配置
    config_suggestion = {
        'size_threshold': float(suggested_threshold),
        'size_threshold_75p': float(threshold_75),
        'size_threshold_90p': float(threshold_90),
        'small_count': int(len(small_areas)),
        'large_count': int(len(large_areas)),
        'small_ratio': float(len(small_areas) / len(areas_relative)),
        'large_ratio': float(len(large_areas) / len(areas_relative)),
    }
    
    print(f"\n{'='*60}")
    print(f"配置建议")
    print(f"{'='*60}")
    print(f"在配置文件中设置:")
    print(f"  DualBranchMatcher:")
    print(f"    size_threshold: {suggested_threshold:.6f}")
    print(f"")
    print(f"小目标占比: {config_suggestion['small_ratio']*100:.1f}%")
    print(f"大目标占比: {config_suggestion['large_ratio']*100:.1f}%")
    
    return config_suggestion


def main():
    parser = argparse.ArgumentParser(description='分析数据集目标尺度分布')
    parser.add_argument('--ann_file', type=str, required=True, help='COCO格式标注文件路径')
    parser.add_argument('--output_dir', type=str, default='./visualizations/scale_analysis', 
                       help='输出目录')
    args = parser.parse_args()
    
    analyze_dataset(args.ann_file, args.output_dir)


if __name__ == '__main__':
    main()

