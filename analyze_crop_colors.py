#!/usr/bin/env python3
"""
分析裁剪目标的色彩分布
结合HSV和植被指数分析绿色植株的色彩范围
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def rgb_to_hsv_array(rgb_array):
    """将RGB数组转换为HSV数组"""
    rgb_normalized = rgb_array.astype(np.float32) / 255.0
    r, g, b = rgb_normalized[..., 0], rgb_normalized[..., 1], rgb_normalized[..., 2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val
    
    # Value
    v = max_val
    
    # Saturation
    s = np.where(max_val > 0, delta / (max_val + 1e-8), 0)
    
    # Hue
    h = np.zeros_like(max_val)
    
    mask_r = (max_val == r) & (delta > 0)
    mask_g = (max_val == g) & (delta > 0)
    mask_b = (max_val == b) & (delta > 0)
    
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-8)) % 6)
    h[mask_g] = 60 * (2 + (b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-8))
    h[mask_b] = 60 * (4 + (r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-8))
    
    return np.stack([h, s * 100, v * 100], axis=-1)  # H: 0-360, S: 0-100, V: 0-100

def calculate_vegetation_indices(rgb_array):
    """计算植被指数"""
    rgb_float = rgb_array.astype(np.float32)
    r, g, b = rgb_float[..., 0], rgb_float[..., 1], rgb_float[..., 2]
    
    # ExG: 过度绿色指数
    exg = 2 * g - r - b
    
    # VARI: 可见光植被指数
    vari = np.where((g + r - b) != 0, (g - r) / (g + r - b + 1e-8), 0)
    
    # GCI: 绿色植被指数 (G/B ratio)
    gci = np.where(b > 0, g / (b + 1e-8), 0)
    
    # 绿色主导度 (Green dominance)
    green_dominance = g - np.maximum(r, b)
    
    return {
        'ExG': exg,
        'VARI': vari,
        'GCI': gci,
        'Green_Dominance': green_dominance
    }

def create_vegetation_mask(hsv_array, veg_indices, method='combined'):
    """创建植被掩码（区分前景植被和背景）"""
    h, s, v = hsv_array[..., 0], hsv_array[..., 1], hsv_array[..., 2]
    
    # 绿色范围 (H: 35-85, S: 20-100, V: 20-100)
    green_mask = (h >= 35) & (h <= 85) & (s >= 20) & (v >= 20)
    
    # 结合ExG
    exg_mask = veg_indices['ExG'] > 20
    
    # 组合掩码
    if method == 'combined':
        return green_mask & exg_mask
    elif method == 'hsv':
        return green_mask
    elif method == 'exg':
        return exg_mask
    return green_mask

def analyze_single_crop(img_path):
    """分析单个裁剪图片"""
    img = Image.open(img_path).convert('RGB')
    rgb_array = np.array(img)
    
    # 转换为HSV
    hsv_array = rgb_to_hsv_array(rgb_array)
    
    # 计算植被指数
    veg_indices = calculate_vegetation_indices(rgb_array)
    
    # 创建植被掩码
    veg_mask = create_vegetation_mask(hsv_array, veg_indices)
    
    # 提取前景（植被）和背景的HSV值
    h, s, v = hsv_array[..., 0], hsv_array[..., 1], hsv_array[..., 2]
    
    fg_pixels = veg_mask.sum()
    total_pixels = veg_mask.size
    fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0
    
    result = {
        'file': img_path.name,
        'size': rgb_array.shape[:2],
        'fg_ratio': fg_ratio,
        # 全图统计
        'h_mean': h.mean(), 'h_std': h.std(),
        's_mean': s.mean(), 's_std': s.std(),
        'v_mean': v.mean(), 'v_std': v.std(),
        # 前景统计
        'fg_h_mean': h[veg_mask].mean() if fg_pixels > 0 else 0,
        'fg_h_std': h[veg_mask].std() if fg_pixels > 0 else 0,
        'fg_s_mean': s[veg_mask].mean() if fg_pixels > 0 else 0,
        'fg_s_std': s[veg_mask].std() if fg_pixels > 0 else 0,
        'fg_v_mean': v[veg_mask].mean() if fg_pixels > 0 else 0,
        'fg_v_std': v[veg_mask].std() if fg_pixels > 0 else 0,
        # 植被指数统计
        'exg_mean': veg_indices['ExG'].mean(),
        'exg_std': veg_indices['ExG'].std(),
        'vari_mean': np.clip(veg_indices['VARI'], -1, 1).mean(),
        'gci_mean': np.clip(veg_indices['GCI'], 0, 5).mean(),
    }
    
    return result, rgb_array, hsv_array, veg_mask, veg_indices

def analyze_all_crops(crop_dir, sample_size=500):
    """分析所有裁剪图片"""
    crop_dir = Path(crop_dir)
    image_files = list(crop_dir.glob('*.jpg')) + list(crop_dir.glob('*.png'))
    
    print(f"Found {len(image_files)} cropped images")
    
    # 随机采样
    if len(image_files) > sample_size:
        image_files = random.sample(image_files, sample_size)
        print(f"Sampling {sample_size} images for analysis")
    
    all_results = []
    all_h_fg, all_s_fg, all_v_fg = [], [], []
    all_h_bg, all_s_bg, all_v_bg = [], [], []
    all_exg, all_vari, all_gci = [], [], []
    
    for img_path in tqdm(image_files, desc="Analyzing crops"):
        try:
            result, rgb_array, hsv_array, veg_mask, veg_indices = analyze_single_crop(img_path)
            all_results.append(result)
            
            h, s, v = hsv_array[..., 0].flatten(), hsv_array[..., 1].flatten(), hsv_array[..., 2].flatten()
            veg_mask_flat = veg_mask.flatten()
            
            # 收集前景和背景HSV值（采样以节省内存）
            sample_idx = np.random.choice(len(h), min(1000, len(h)), replace=False)
            
            fg_idx = sample_idx[veg_mask_flat[sample_idx]]
            bg_idx = sample_idx[~veg_mask_flat[sample_idx]]
            
            if len(fg_idx) > 0:
                all_h_fg.extend(h[fg_idx].tolist())
                all_s_fg.extend(s[fg_idx].tolist())
                all_v_fg.extend(v[fg_idx].tolist())
            
            if len(bg_idx) > 0:
                all_h_bg.extend(h[bg_idx].tolist())
                all_s_bg.extend(s[bg_idx].tolist())
                all_v_bg.extend(v[bg_idx].tolist())
            
            # 收集植被指数
            all_exg.extend(veg_indices['ExG'].flatten()[sample_idx].tolist())
            all_vari.extend(np.clip(veg_indices['VARI'].flatten()[sample_idx], -1, 1).tolist())
            all_gci.extend(np.clip(veg_indices['GCI'].flatten()[sample_idx], 0, 5).tolist())
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return all_results, {
        'h_fg': np.array(all_h_fg), 's_fg': np.array(all_s_fg), 'v_fg': np.array(all_v_fg),
        'h_bg': np.array(all_h_bg), 's_bg': np.array(all_s_bg), 'v_bg': np.array(all_v_bg),
        'exg': np.array(all_exg), 'vari': np.array(all_vari), 'gci': np.array(all_gci)
    }

def visualize_color_distribution(pixel_data, output_dir):
    """可视化色彩分布"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # HSV分布 - 前景 vs 背景
    # Hue分布
    ax = axes[0, 0]
    if len(pixel_data['h_fg']) > 0:
        ax.hist(pixel_data['h_fg'], bins=72, range=(0, 360), alpha=0.7, label='Foreground (Plant)', color='green', density=True)
    if len(pixel_data['h_bg']) > 0:
        ax.hist(pixel_data['h_bg'], bins=72, range=(0, 360), alpha=0.5, label='Background', color='gray', density=True)
    ax.set_xlabel('Hue (0-360)')
    ax.set_ylabel('Density')
    ax.set_title('Hue Distribution: Foreground vs Background')
    ax.legend()
    ax.axvspan(35, 85, alpha=0.2, color='green', label='Green range')
    
    # Saturation分布
    ax = axes[0, 1]
    if len(pixel_data['s_fg']) > 0:
        ax.hist(pixel_data['s_fg'], bins=50, range=(0, 100), alpha=0.7, label='Foreground (Plant)', color='green', density=True)
    if len(pixel_data['s_bg']) > 0:
        ax.hist(pixel_data['s_bg'], bins=50, range=(0, 100), alpha=0.5, label='Background', color='gray', density=True)
    ax.set_xlabel('Saturation (0-100)')
    ax.set_ylabel('Density')
    ax.set_title('Saturation Distribution: Foreground vs Background')
    ax.legend()
    
    # Value分布
    ax = axes[0, 2]
    if len(pixel_data['v_fg']) > 0:
        ax.hist(pixel_data['v_fg'], bins=50, range=(0, 100), alpha=0.7, label='Foreground (Plant)', color='green', density=True)
    if len(pixel_data['v_bg']) > 0:
        ax.hist(pixel_data['v_bg'], bins=50, range=(0, 100), alpha=0.5, label='Background', color='gray', density=True)
    ax.set_xlabel('Value (0-100)')
    ax.set_ylabel('Density')
    ax.set_title('Value Distribution: Foreground vs Background')
    ax.legend()
    
    # H-S 2D直方图
    ax = axes[1, 0]
    if len(pixel_data['h_fg']) > 0:
        ax.hist2d(pixel_data['h_fg'], pixel_data['s_fg'], bins=[72, 50], range=[[0, 360], [0, 100]], cmap='Greens')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_title('H-S Distribution (Foreground/Plant)')
    ax.axvline(x=35, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=85, color='red', linestyle='--', alpha=0.7)
    
    # H-V 2D直方图
    ax = axes[1, 1]
    if len(pixel_data['h_fg']) > 0:
        ax.hist2d(pixel_data['h_fg'], pixel_data['v_fg'], bins=[72, 50], range=[[0, 360], [0, 100]], cmap='Greens')
    ax.set_xlabel('Hue')
    ax.set_ylabel('Value')
    ax.set_title('H-V Distribution (Foreground/Plant)')
    ax.axvline(x=35, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=85, color='red', linestyle='--', alpha=0.7)
    
    # S-V 2D直方图
    ax = axes[1, 2]
    if len(pixel_data['h_fg']) > 0:
        ax.hist2d(pixel_data['s_fg'], pixel_data['v_fg'], bins=[50, 50], range=[[0, 100], [0, 100]], cmap='Greens')
    ax.set_xlabel('Saturation')
    ax.set_ylabel('Value')
    ax.set_title('S-V Distribution (Foreground/Plant)')
    
    # 植被指数分布
    ax = axes[2, 0]
    ax.hist(pixel_data['exg'], bins=100, range=(-100, 200), alpha=0.7, color='green', density=True)
    ax.set_xlabel('ExG (Excess Green)')
    ax.set_ylabel('Density')
    ax.set_title('ExG Distribution')
    ax.axvline(x=20, color='red', linestyle='--', label='Threshold=20')
    ax.legend()
    
    ax = axes[2, 1]
    ax.hist(pixel_data['vari'], bins=100, range=(-1, 1), alpha=0.7, color='green', density=True)
    ax.set_xlabel('VARI')
    ax.set_ylabel('Density')
    ax.set_title('VARI Distribution')
    
    ax = axes[2, 2]
    ax.hist(pixel_data['gci'], bins=100, range=(0, 5), alpha=0.7, color='green', density=True)
    ax.set_xlabel('GCI (G/B Ratio)')
    ax.set_ylabel('Density')
    ax.set_title('GCI Distribution')
    ax.axvline(x=1.0, color='red', linestyle='--', label='G=B')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'color_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'color_distribution.png'}")

def visualize_sample_crops(crop_dir, output_dir, num_samples=16):
    """可视化样本裁剪及其分析结果"""
    crop_dir = Path(crop_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(crop_dir.glob('*.jpg')) + list(crop_dir.glob('*.png'))
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    n_cols = 4
    n_rows = (num_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, img_path in enumerate(samples):
        result, rgb_array, hsv_array, veg_mask, veg_indices = analyze_single_crop(img_path)
        
        ax = axes[idx]
        
        # 创建叠加显示
        overlay = rgb_array.copy()
        overlay[~veg_mask] = (overlay[~veg_mask] * 0.3).astype(np.uint8)  # 背景变暗
        
        ax.imshow(overlay)
        ax.set_title(f"FG: {result['fg_ratio']*100:.1f}%\nH:{result['fg_h_mean']:.0f}±{result['fg_h_std']:.0f}", fontsize=9)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sample Crops with Vegetation Mask Overlay', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_crops_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'sample_crops_analysis.png'}")

def print_color_range_summary(pixel_data, results):
    """打印色彩范围总结"""
    print("\n" + "="*70)
    print("GREEN PLANT COLOR RANGE ANALYSIS SUMMARY")
    print("="*70)
    
    # 计算前景HSV统计
    if len(pixel_data['h_fg']) > 0:
        h_fg = pixel_data['h_fg']
        s_fg = pixel_data['s_fg']
        v_fg = pixel_data['v_fg']
        
        print("\n[FOREGROUND (PLANT) HSV RANGE]")
        print("-"*50)
        print(f"  Hue:        Mean={h_fg.mean():.1f}, Std={h_fg.std():.1f}")
        print(f"              P5={np.percentile(h_fg, 5):.1f}, P95={np.percentile(h_fg, 95):.1f}")
        print(f"              Recommended range: [{np.percentile(h_fg, 5):.0f}, {np.percentile(h_fg, 95):.0f}]")
        print()
        print(f"  Saturation: Mean={s_fg.mean():.1f}, Std={s_fg.std():.1f}")
        print(f"              P5={np.percentile(s_fg, 5):.1f}, P95={np.percentile(s_fg, 95):.1f}")
        print(f"              Recommended range: [{np.percentile(s_fg, 5):.0f}, {np.percentile(s_fg, 95):.0f}]")
        print()
        print(f"  Value:      Mean={v_fg.mean():.1f}, Std={v_fg.std():.1f}")
        print(f"              P5={np.percentile(v_fg, 5):.1f}, P95={np.percentile(v_fg, 95):.1f}")
        print(f"              Recommended range: [{np.percentile(v_fg, 5):.0f}, {np.percentile(v_fg, 95):.0f}]")
    
    if len(pixel_data['h_bg']) > 0:
        h_bg = pixel_data['h_bg']
        s_bg = pixel_data['s_bg']
        v_bg = pixel_data['v_bg']
        
        print("\n[BACKGROUND HSV RANGE]")
        print("-"*50)
        print(f"  Hue:        Mean={h_bg.mean():.1f}, Std={h_bg.std():.1f}")
        print(f"  Saturation: Mean={s_bg.mean():.1f}, Std={s_bg.std():.1f}")
        print(f"  Value:      Mean={v_bg.mean():.1f}, Std={v_bg.std():.1f}")
    
    print("\n[VEGETATION INDICES THRESHOLDS]")
    print("-"*50)
    exg = pixel_data['exg']
    vari = pixel_data['vari']
    gci = pixel_data['gci']
    
    print(f"  ExG:  Mean={exg.mean():.1f}, Std={exg.std():.1f}")
    print(f"        P25={np.percentile(exg, 25):.1f}, Median={np.median(exg):.1f}, P75={np.percentile(exg, 75):.1f}")
    print(f"        Recommended threshold: > {np.percentile(exg, 25):.0f}")
    print()
    print(f"  VARI: Mean={vari.mean():.3f}, Std={vari.std():.3f}")
    print(f"        Recommended range: > {np.percentile(vari, 25):.2f}")
    print()
    print(f"  GCI:  Mean={gci.mean():.2f}, Std={gci.std():.2f}")
    print(f"        Recommended threshold: > {np.percentile(gci, 25):.2f}")
    
    # 前景比例统计
    fg_ratios = [r['fg_ratio'] for r in results]
    print("\n[FOREGROUND RATIO STATISTICS]")
    print("-"*50)
    print(f"  Mean foreground ratio: {np.mean(fg_ratios)*100:.1f}%")
    print(f"  Median: {np.median(fg_ratios)*100:.1f}%")
    print(f"  P25-P75: {np.percentile(fg_ratios, 25)*100:.1f}% - {np.percentile(fg_ratios, 75)*100:.1f}%")
    
    print("\n" + "="*70)
    print("RECOMMENDED HSV RANGE FOR GREEN PLANT DETECTION:")
    print("="*70)
    if len(pixel_data['h_fg']) > 0:
        h_fg = pixel_data['h_fg']
        s_fg = pixel_data['s_fg']
        v_fg = pixel_data['v_fg']
        print(f"  H: [{np.percentile(h_fg, 5):.0f}, {np.percentile(h_fg, 95):.0f}]")
        print(f"  S: [{max(10, np.percentile(s_fg, 5)):.0f}, {np.percentile(s_fg, 95):.0f}]")
        print(f"  V: [{max(10, np.percentile(v_fg, 5)):.0f}, {np.percentile(v_fg, 95):.0f}]")
    print("="*70)

def main():
    crop_dir = 'DJI_dataset/cropped_bboxes/target'
    output_dir = 'vegetation_analysis/crop_color_analysis'
    
    print("Analyzing cropped target color distribution...")
    print(f"Input: {crop_dir}")
    print(f"Output: {output_dir}")
    
    # 分析所有裁剪
    results, pixel_data = analyze_all_crops(crop_dir, sample_size=1000)
    
    # 可视化
    visualize_color_distribution(pixel_data, output_dir)
    visualize_sample_crops(crop_dir, output_dir, num_samples=16)
    
    # 打印总结
    print_color_range_summary(pixel_data, results)

if __name__ == '__main__':
    main()

