#!/usr/bin/env python3
"""
边缘特征提取和可视化脚本
读取一张图片，使用Canny算子提取边缘特征并进行可视化
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import argparse


# ============================================================
# 1. 图像预处理
# ============================================================

def load_image(img_path):
    """加载图片"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"无法加载图片: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(rgb_array):
    """图像预处理：去噪、增强"""
    # 转灰度
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # CLAHE对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    return gray, blurred, enhanced


# ============================================================
# 2. Canny边缘检测（支持自适应阈值）
# ============================================================

def detect_edges_canny(gray, low_threshold=50, high_threshold=150, use_auto_threshold=False):
    """
    Canny边缘检测
    
    Args:
        gray: 灰度图像
        low_threshold: 低阈值（用于边缘连接）
        high_threshold: 高阈值（用于初始边缘检测）
        use_auto_threshold: 是否使用自适应阈值（基于图像统计）
    
    Returns:
        edges: 边缘图像
    """
    if use_auto_threshold:
        # 使用Otsu方法自动计算阈值
        # 或者使用图像统计信息
        sigma = 0.33
        median = np.median(gray)
        low_threshold = int(max(0, (1.0 - sigma) * median))
        high_threshold = int(min(255, (1.0 + sigma) * median))
        print(f"  自动计算阈值: low={low_threshold}, high={high_threshold}")
    
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges, low_threshold, high_threshold


# ============================================================
# 3. 边缘特征提取
# ============================================================

def extract_edge_features(edges, mask=None):
    """从边缘图像提取特征"""
    features = {}
    
    # 如果有掩码，只在掩码区域内分析
    if mask is not None:
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    else:
        edges_masked = edges
    
    # 边缘像素数
    edge_pixels = np.sum(edges_masked > 0)
    total_pixels = edges_masked.size
    features['edge_density'] = edge_pixels / total_pixels if total_pixels > 0 else 0
    
    # 边缘强度统计
    edge_values = edges_masked[edges_masked > 0]
    if len(edge_values) > 0:
        features['edge_mean'] = np.mean(edge_values)
        features['edge_std'] = np.std(edge_values)
        features['edge_max'] = np.max(edge_values)
        features['edge_min'] = np.min(edge_values)
        features['edge_median'] = np.median(edge_values)
    else:
        features['edge_mean'] = 0
        features['edge_std'] = 0
        features['edge_max'] = 0
        features['edge_min'] = 0
        features['edge_median'] = 0
    
    # 边缘连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges_masked, connectivity=8)
    features['num_edge_components'] = num_labels - 1  # 减去背景
    if num_labels > 1:
        component_sizes = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
        features['avg_component_size'] = np.mean(component_sizes)
        features['max_component_size'] = np.max(component_sizes)
    else:
        features['avg_component_size'] = 0
        features['max_component_size'] = 0
    
    return features


# ============================================================
# 4. 可视化函数
# ============================================================

def visualize_edge_features(img_path, output_path=None, 
                           low_threshold=50, high_threshold=150,
                           use_auto_threshold=False, show_stats=True):
    """
    可视化边缘特征
    
    Args:
        img_path: 输入图片路径
        output_path: 输出图片路径（可选）
        low_threshold: Canny低阈值
        high_threshold: Canny高阈值
        use_auto_threshold: 是否使用自适应阈值
        show_stats: 是否显示统计信息
    """
    print(f"\n{'='*60}")
    print("边缘特征提取和可视化")
    print(f"{'='*60}")
    print(f"输入图片: {img_path}")
    print(f"阈值设置: low={low_threshold}, high={high_threshold}")
    print(f"自适应阈值: {use_auto_threshold}")
    
    # 1. 加载和预处理图片
    print("\n[1/4] 加载和预处理图片...")
    rgb_array = load_image(img_path)
    gray, blurred, enhanced = preprocess_image(rgb_array)
    print(f"   图片尺寸: {rgb_array.shape}")
    print(f"   灰度图范围: [{gray.min()}, {gray.max()}]")
    
    # 2. Canny边缘检测
    print("\n[2/4] Canny边缘检测...")
    edges, actual_low, actual_high = detect_edges_canny(
        enhanced, low_threshold, high_threshold, use_auto_threshold
    )
    print(f"   边缘像素数: {np.sum(edges > 0)}")
    print(f"   边缘密度: {np.sum(edges > 0) / edges.size * 100:.2f}%")
    
    # 3. 提取边缘特征
    print("\n[3/4] 提取边缘特征...")
    features = extract_edge_features(edges)
    
    # 打印特征统计
    print("\n   边缘特征统计:")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"   {key:25s}: {value:.4f}")
        else:
            print(f"   {key:25s}: {value}")
    
    # 4. 可视化
    print("\n[4/4] 生成可视化...")
    
    # 创建图形布局
    if show_stats:
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 第一行：原图、灰度图、增强图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_array)
    ax1.set_title('原始图片', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gray, cmap='gray')
    ax2.set_title('灰度图', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(enhanced, cmap='gray')
    ax3.set_title('CLAHE增强后', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 第二行：边缘检测结果
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(edges, cmap='gray')
    ax4.set_title(f'Canny边缘检测\n(阈值: {actual_low}, {actual_high})', 
                  fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 边缘叠加在原图上
    ax5 = fig.add_subplot(gs[1, 1])
    overlay = rgb_array.copy()
    # 将边缘标记为红色
    overlay[edges > 0] = [255, 0, 0]
    ax5.imshow(overlay)
    ax5.set_title('边缘叠加（红色）', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 边缘叠加（半透明）
    ax6 = fig.add_subplot(gs[1, 2])
    overlay_alpha = rgb_array.copy()
    edge_mask = edges > 0
    overlay_alpha[edge_mask] = overlay_alpha[edge_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    ax6.imshow(overlay_alpha)
    ax6.set_title('边缘叠加（半透明）', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # 第三行：特征统计和直方图
    if show_stats:
        # 特征统计文本
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.axis('off')
        stats_text = "边缘特征统计:\n\n"
        for key, value in features.items():
            if isinstance(value, float):
                stats_text += f"{key:25s}: {value:.4f}\n"
            else:
                stats_text += f"{key:25s}: {value}\n"
        ax7.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax7.transAxes)
        ax7.set_title('特征统计', fontsize=12, fontweight='bold')
        
        # 边缘强度直方图
        ax8 = fig.add_subplot(gs[2, 1])
        edge_values = edges[edges > 0]
        if len(edge_values) > 0:
            ax8.hist(edge_values, bins=50, edgecolor='black', alpha=0.7, color='red')
            ax8.axvline(features['edge_mean'], color='blue', linestyle='--', 
                       linewidth=2, label=f"均值: {features['edge_mean']:.1f}")
            ax8.axvline(features['edge_median'], color='green', linestyle='--', 
                       linewidth=2, label=f"中位数: {features['edge_median']:.1f}")
            ax8.set_xlabel('边缘强度', fontsize=10)
            ax8.set_ylabel('频数', fontsize=10)
            ax8.set_title('边缘强度分布', fontsize=12, fontweight='bold')
            ax8.legend(fontsize=9)
            ax8.grid(True, alpha=0.3)
        
        # 边缘密度可视化（热力图）
        ax9 = fig.add_subplot(gs[2, 2])
        # 计算局部边缘密度（使用滑动窗口）
        kernel_size = 20
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        edge_density_map = cv2.filter2D(edges.astype(np.float32), -1, kernel)
        im = ax9.imshow(edge_density_map, cmap='hot', interpolation='bilinear')
        ax9.set_title('边缘密度热力图', fontsize=12, fontweight='bold')
        ax9.axis('off')
        plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
        
        # 不同阈值对比（如果使用固定阈值）
        if not use_auto_threshold:
            ax10 = fig.add_subplot(gs[2, 3])
            # 尝试不同的阈值组合
            threshold_pairs = [
                (low_threshold * 0.5, high_threshold * 0.5, "低阈值"),
                (low_threshold, high_threshold, "当前阈值"),
                (low_threshold * 1.5, high_threshold * 1.5, "高阈值"),
            ]
            
            comparison_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            
            for i, (low, high, label) in enumerate(threshold_pairs):
                test_edges, _, _ = detect_edges_canny(enhanced, int(low), int(high), False)
                comparison_img[test_edges > 0] = colors[i]
            
            ax10.imshow(comparison_img)
            ax10.set_title('不同阈值对比\n(红:低, 绿:中, 蓝:高)', fontsize=12, fontweight='bold')
            ax10.axis('off')
    
    # 添加总标题
    fig.suptitle(f'边缘特征提取和可视化 - {Path(img_path).name}', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 可视化结果已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return features, edges


# ============================================================
# 5. 参数调优辅助函数
# ============================================================

def compare_thresholds(img_path, output_path=None):
    """比较不同阈值设置的效果"""
    print(f"\n{'='*60}")
    print("阈值参数对比分析")
    print(f"{'='*60}")
    
    rgb_array = load_image(img_path)
    gray, blurred, enhanced = preprocess_image(rgb_array)
    
    # 测试不同的阈值组合
    threshold_configs = [
        (30, 100, "低阈值 (30, 100)"),
        (50, 150, "中阈值 (50, 150)"),
        (80, 200, "高阈值 (80, 200)"),
        (100, 250, "很高阈值 (100, 250)"),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (low, high, label) in enumerate(threshold_configs):
        edges, _, _ = detect_edges_canny(enhanced, low, high, False)
        features = extract_edge_features(edges)
        
        # 显示边缘图
        axes[idx].imshow(edges, cmap='gray')
        axes[idx].set_title(f'{label}\n密度: {features["edge_density"]:.3f}', 
                           fontsize=11, fontweight='bold')
        axes[idx].axis('off')
        
        # 显示叠加图
        overlay = rgb_array.copy()
        overlay[edges > 0] = [255, 0, 0]
        axes[idx + 4].imshow(overlay)
        axes[idx + 4].set_title(f'{label}\n组件数: {features["num_edge_components"]}', 
                               fontsize=11, fontweight='bold')
        axes[idx + 4].axis('off')
    
    plt.suptitle('不同阈值设置对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 对比结果已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='边缘特征提取和可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用（默认阈值）
  python visualize_edge_features.py --image path/to/image.jpg
  
  # 指定阈值
  python visualize_edge_features.py --image path/to/image.jpg --low 30 --high 100
  
  # 使用自适应阈值
  python visualize_edge_features.py --image path/to/image.jpg --auto-threshold
  
  # 比较不同阈值
  python visualize_edge_features.py --image path/to/image.jpg --compare-thresholds
        """
    )
    
    parser.add_argument('--image', type=str, default='DJI_dataset/images/coco_dataset_1_DJI_20250626101954_0001_patch_0000_0000.jpg' ,
                       help='输入图片路径')
    parser.add_argument('--output', type=str, default='vegetation_analysis/canny_comparison',
                       help='输出图片路径（默认：输入图片名_edge_features.png）')
    parser.add_argument('--low', type=int, default=50,
                       help='Canny低阈值（默认：50）')
    parser.add_argument('--high', type=int, default=150,
                       help='Canny高阈值（默认：150）')
    parser.add_argument('--auto-threshold', action='store_true',default=True,
                       help='使用自适应阈值（基于图像统计）')
    parser.add_argument('--no-stats', action='store_true',
                       help='不显示统计信息')
    parser.add_argument('--compare-thresholds', action='store_true',default=False,
                       help='比较不同阈值设置的效果')
    
    args = parser.parse_args()
    
    # 检查输入文件
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"❌ 错误: 图片文件不存在: {img_path}")
        exit(1)
    
    # 设置输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = img_path.parent / f"{img_path.stem}_edge_features.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.compare_thresholds:
            # 阈值对比模式
            compare_output = img_path.parent / f"{img_path.stem}_threshold_comparison.png"
            compare_thresholds(img_path, compare_output)
        else:
            # 正常可视化模式
            features, edges = visualize_edge_features(
                img_path=img_path,
                output_path=output_path,
                low_threshold=args.low,
                high_threshold=args.high,
                use_auto_threshold=args.auto_threshold,
                show_stats=not args.no_stats
            )
            
            print(f"\n{'='*60}")
            print("✅ 处理完成！")
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

