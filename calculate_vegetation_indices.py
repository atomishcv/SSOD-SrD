"""
基于RGB波段的植被指数计算与可视化
支持多种植被指数：VARI, GCI, CVI, SAVI, ExG
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from pathlib import Path


def load_image(image_path):
    """
    加载图片并转换为RGB格式
    
    Args:
        image_path: 图片路径
        
    Returns:
        image_rgb: RGB格式的图片数组 (H, W, 3)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 使用OpenCV读取图片（BGR格式）
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb


def calculate_vari(r, g, b):
    """
    计算VARI（可变植被指数）
    VARI = (G - R) / (G + R - B)
    
    Args:
        r, g, b: 红、绿、蓝波段
        
    Returns:
        vari: VARI指数
    """
    denominator = g + r - b
    # 避免除零
    denominator = np.where(denominator == 0, 1e-10, denominator)
    vari = (g - r) / denominator
    return vari


def calculate_gci(r, g, b):
    """
    计算GCI（绿色植被指数）
    GCI = G / B
    
    Args:
        r, g, b: 红、绿、蓝波段
        
    Returns:
        gci: GCI指数
    """
    # 避免除零
    b = np.where(b == 0, 1e-10, b)
    gci = g / b
    return gci


def calculate_cvi(r, g, b):
    """
    计算CVI（色度植被指数）
    CVI = (G * G) / (R * R * B)
    
    Args:
        r, g, b: 红、绿、蓝波段
        
    Returns:
        cvi: CVI指数
    """
    # 避免除零
    denominator = r * r * b
    denominator = np.where(denominator == 0, 1e-10, denominator)
    cvi = (g * g) / denominator
    return cvi


def calculate_savi(r, g, b, l=0.5):
    """
    计算SAVI（简化植被指数，RGB版本）
    SAVI = (G - R) / (G + R + L)
    
    Args:
        r, g, b: 红、绿、蓝波段
        l: 调节因子，默认0.5
        
    Returns:
        savi: SAVI指数
    """
    denominator = g + r + l
    savi = (g - r) / denominator
    return savi


def calculate_exg(r, g, b):
    """
    计算ExG（绿色增强指数）
    ExG = 2 * G - R - B
    
    Args:
        r, g, b: 红、绿、蓝波段
        
    Returns:
        exg: ExG指数
    """
    exg = 2 * g - r - b
    return exg


def calculate_all_indices(image_rgb):
    """
    计算所有植被指数
    
    Args:
        image_rgb: RGB格式的图片数组 (H, W, 3)
        
    Returns:
        indices_dict: 包含所有植被指数的字典
    """
    # 分离RGB波段并转换为float类型
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    
    # 计算各种植被指数
    indices_dict = {
        'VARI': calculate_vari(r, g, b),
        'GCI': calculate_gci(r, g, b),
        'CVI': calculate_cvi(r, g, b),
        'SAVI': calculate_savi(r, g, b),
        'EXG': calculate_exg(r, g, b)
    }
    
    return indices_dict


def normalize_index(index, method='percentile', lower=2, upper=98):
    """
    归一化植被指数用于可视化
    
    Args:
        index: 植被指数数组
        method: 归一化方法 ('percentile' 或 'minmax')
        lower: 下百分位数（用于percentile方法）
        upper: 上百分位数（用于percentile方法）
        
    Returns:
        normalized: 归一化后的指数 [0, 1]
    """
    if method == 'percentile':
        # 使用百分位数裁剪
        p_lower = np.percentile(index, lower)
        p_upper = np.percentile(index, upper)
        clipped = np.clip(index, p_lower, p_upper)
        normalized = (clipped - p_lower) / (p_upper - p_lower + 1e-10)
    else:
        # 使用最小最大值归一化
        min_val = np.nanmin(index)
        max_val = np.nanmax(index)
        normalized = (index - min_val) / (max_val - min_val + 1e-10)
    
    return np.clip(normalized, 0, 1)


def visualize_indices(image_rgb, indices_dict, save_path=None):
    """
    可视化原始图片和所有植被指数
    
    Args:
        image_rgb: 原始RGB图片
        indices_dict: 包含所有植被指数的字典
        save_path: 保存路径（可选）
    """
    n_indices = len(indices_dict)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Display original image
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image (RGB)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Display vegetation indices
    index_names = ['VARI', 'GCI', 'CVI', 'SAVI', 'EXG']
    colors = ['RdYlGn', 'Greens', 'YlGn', 'viridis', 'terrain']
    
    for idx, (name, color) in enumerate(zip(index_names, colors), start=1):
        index = indices_dict[name]
        
        # Normalize for visualization
        normalized = normalize_index(index, method='percentile')
        
        # Display
        im = axes[idx].imshow(normalized, cmap=color, vmin=0, vmax=1)
        axes[idx].set_title(f'{name} Vegetation Index', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.close()  # 关闭图形以释放内存


def convert_rgb_to_hsv(image_rgb):
    """
    Convert RGB image to HSV color space
    
    Args:
        image_rgb: RGB image array (H, W, 3), values in [0, 255]
        
    Returns:
        image_hsv: HSV image array (H, W, 3)
            - H (Hue): [0, 179] in OpenCV, [0, 360] in standard HSV
            - S (Saturation): [0, 255]
            - V (Value/Brightness): [0, 255]
    """
    # Convert to float and normalize to [0, 1] for OpenCV
    image_rgb_float = image_rgb.astype(np.float32) / 255.0
    
    # Convert RGB to HSV using OpenCV
    image_hsv = cv2.cvtColor(image_rgb_float, cv2.COLOR_RGB2HSV)
    
    # Scale H to [0, 179] (OpenCV format), S and V to [0, 255]
    image_hsv[:, :, 0] = image_hsv[:, :, 0]  # H is already in [0, 179] range
    image_hsv[:, :, 1] = image_hsv[:, :, 1] * 255  # S: [0, 1] -> [0, 255]
    image_hsv[:, :, 2] = image_hsv[:, :, 2] * 255  # V: [0, 1] -> [0, 255]
    
    return image_hsv.astype(np.uint8)


def visualize_hsv(image_rgb, image_hsv, save_path=None):
    """
    Visualize RGB image and HSV channels
    
    Args:
        image_rgb: Original RGB image
        image_hsv: HSV image array
        save_path: Path to save the visualization (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # First row: RGB channels
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original RGB Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # RGB channels
    axes[0, 1].imshow(image_rgb[:, :, 0], cmap='Reds')
    axes[0, 1].set_title('Red Channel', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].imshow(image_rgb[:, :, 0], cmap='Reds'), ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    axes[0, 2].imshow(image_rgb[:, :, 1], cmap='Greens')
    axes[0, 2].set_title('Green Channel', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(axes[0, 2].imshow(image_rgb[:, :, 1], cmap='Greens'), ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Second row: HSV channels
    # Convert HSV back to RGB for display (full color HSV representation)
    hsv_display = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    axes[1, 0].imshow(hsv_display)
    axes[1, 0].set_title('HSV Color Space (converted to RGB for display)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # H (Hue) channel - use hsv colormap
    h_channel = image_hsv[:, :, 0]
    im_h = axes[1, 1].imshow(h_channel, cmap='hsv', vmin=0, vmax=179)
    axes[1, 1].set_title('Hue Channel (H)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im_h, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # S (Saturation) channel
    s_channel = image_hsv[:, :, 1]
    im_s = axes[1, 2].imshow(s_channel, cmap='gray', vmin=0, vmax=255)
    axes[1, 2].set_title('Saturation Channel (S)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im_s, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"HSV visualization saved to: {save_path}")
    
    plt.close()
    
    # Create additional visualization for V channel and combined HSV
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    # V (Value/Brightness) channel
    v_channel = image_hsv[:, :, 2]
    im_v = axes2[0].imshow(v_channel, cmap='gray', vmin=0, vmax=255)
    axes2[0].set_title('Value/Brightness Channel (V)', fontsize=14, fontweight='bold')
    axes2[0].axis('off')
    plt.colorbar(im_v, ax=axes2[0], fraction=0.046, pad=0.04)
    
    # Blue channel from RGB
    axes2[1].imshow(image_rgb[:, :, 2], cmap='Blues')
    axes2[1].set_title('Blue Channel (RGB)', fontsize=14, fontweight='bold')
    axes2[1].axis('off')
    plt.colorbar(axes2[1].imshow(image_rgb[:, :, 2], cmap='Blues'), ax=axes2[1], fraction=0.046, pad=0.04)
    
    # Combined HSV visualization
    axes2[2].imshow(hsv_display)
    axes2[2].set_title('HSV Color Space', fontsize=14, fontweight='bold')
    axes2[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path_v = str(save_path).replace('.png', '_v_channel.png')
        plt.savefig(save_path_v, dpi=300, bbox_inches='tight')
        print(f"HSV V channel visualization saved to: {save_path_v}")
    
    plt.close()


def print_hsv_statistics(image_hsv):
    """
    Print statistics for HSV channels
    
    Args:
        image_hsv: HSV image array
    """
    print("\n" + "="*60)
    print("HSV Color Space Statistics")
    print("="*60)
    
    channel_names = ['Hue (H)', 'Saturation (S)', 'Value (V)']
    
    for idx, name in enumerate(channel_names):
        channel = image_hsv[:, :, idx]
        valid_channel = channel[~np.isnan(channel)]
        
        if len(valid_channel) > 0:
            print(f"\n{name}:")
            print(f"  Min: {np.min(valid_channel):.2f}")
            print(f"  Max: {np.max(valid_channel):.2f}")
            print(f"  Mean: {np.mean(valid_channel):.2f}")
            print(f"  Median: {np.median(valid_channel):.2f}")
            print(f"  Std: {np.std(valid_channel):.2f}")


def create_vegetation_mask(indices_dict, threshold_method='exg', threshold_value=None):
    """
    根据植被指数创建植被掩膜
    
    Args:
        indices_dict: 包含所有植被指数的字典
        threshold_method: 使用的指数方法 ('exg', 'vari', 'gci', 'cvi', 'savi')
        threshold_value: 阈值（如果为None，则自动计算）
        
    Returns:
        mask: 二值掩膜（True表示植被区域）
    """
    method_upper = threshold_method.upper()
    if method_upper not in indices_dict:
        raise ValueError(f"不支持的指数方法: {threshold_method}。支持的方法: {list(indices_dict.keys())}")
    
    index = indices_dict[method_upper]
    
    if threshold_value is None:
        # 自动计算阈值（使用百分位数方法）
        # 移除异常值（NaN和Inf）
        valid_index = index[~np.isnan(index) & ~np.isinf(index)]
        
        if len(valid_index) == 0:
            # 如果没有有效值，返回全False掩膜
            return np.zeros_like(index, dtype=bool)
        
        if method_upper == 'EXG':
            # ExG可能为负值，使用正值的50百分位数（中位数）
            positive_index = valid_index[valid_index > 0]
            if len(positive_index) > 0:
                threshold_value = np.percentile(positive_index, 50)
            else:
                threshold_value = 0
        elif method_upper == 'VARI' or method_upper == 'SAVI':
            # VARI和SAVI可能为负值，使用75百分位数
            threshold_value = np.percentile(valid_index, 75)
        elif method_upper == 'GCI' or method_upper == 'CVI':
            # GCI和CVI可能有极端值，使用98百分位数裁剪后再计算75百分位数
            p98 = np.percentile(valid_index, 98)
            clipped_index = valid_index[valid_index <= p98]
            if len(clipped_index) > 0:
                threshold_value = np.percentile(clipped_index, 75)
            else:
                threshold_value = np.percentile(valid_index, 75)
        else:
            # 默认使用75百分位数
            threshold_value = np.percentile(valid_index, 75)
    
    mask = index > threshold_value
    return mask


def print_statistics(indices_dict):
    """
    打印各个植被指数的统计信息
    
    Args:
        indices_dict: 包含所有植被指数的字典
    """
    print("\n" + "="*60)
    print("Vegetation Index Statistics")
    print("="*60)
    
    for name, index in indices_dict.items():
        valid_index = index[~np.isnan(index) & ~np.isinf(index)]
        if len(valid_index) > 0:
            print(f"\n{name}:")
            print(f"  Min: {np.min(valid_index):.4f}")
            print(f"  Max: {np.max(valid_index):.4f}")
            print(f"  Mean: {np.mean(valid_index):.4f}")
            print(f"  Median: {np.median(valid_index):.4f}")
            print(f"  Std: {np.std(valid_index):.4f}")
            print(f"  2nd Percentile: {np.percentile(valid_index, 2):.4f}")
            print(f"  98th Percentile: {np.percentile(valid_index, 98):.4f}")


def main():
    """Main function"""
    # Image path
    image_path = "/data/DEIMv2/DJI_dataset/images/coco_dataset_1_DJI_20250626102509_0316_patch_0922_0922.jpg"
    
    print(f"Loading image: {image_path}")
    
    # Setup output directory
    output_dir = Path("/data/DEIMv2") / "vegetation_analysis/003" 
    output_dir.mkdir(exist_ok=True)
    
    # Load image
    try:
        image_rgb = load_image(image_path)
        print(f"Image size: {image_rgb.shape}")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Convert RGB to HSV
    print("\nConverting RGB to HSV color space...")
    image_hsv = convert_rgb_to_hsv(image_rgb)
    print("HSV conversion completed!")
    
    # Print HSV statistics
    print_hsv_statistics(image_hsv)
    
    # Visualize HSV
    print("\nGenerating HSV visualization...")
    hsv_save_path = output_dir / "hsv_visualization.png"
    visualize_hsv(image_rgb, image_hsv, save_path=str(hsv_save_path))
    
    # Calculate all vegetation indices
    print("\nCalculating vegetation indices...")
    indices_dict = calculate_all_indices(image_rgb)
    print("Calculation completed!")
    
    # Print statistics
    print_statistics(indices_dict)
    
    # Visualization
    print("\nGenerating visualization...")
    
    save_path = output_dir / "vegetation_indices_visualization.png"
    visualize_indices(image_rgb, indices_dict, save_path=str(save_path))
    
    # Create mask and overlay visualization for each vegetation index
    print("\nCreating mask and overlay visualization for each vegetation index...")
    
    index_methods = ['VARI', 'GCI', 'CVI', 'SAVI', 'EXG']
    index_colors = {
        'VARI': [0, 255, 0],      # 绿色
        'GCI': [0, 255, 0],       # 绿色
        'CVI': [0, 255, 0],       # 绿色
        'SAVI': [0, 255, 0],      # 绿色
        'EXG': [0, 255, 0]        # 绿色
    }
    
    for method in index_methods:
        try:
            # Create mask
            mask = create_vegetation_mask(indices_dict, threshold_method=method.lower())
            vegetation_ratio = np.sum(mask) / mask.size * 100
            print(f"  {method}: Vegetation Coverage = {vegetation_ratio:.2f}%")
            
            # Create triple-panel visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Left: Original image
            axes[0].imshow(image_rgb)
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Middle: Vegetation mask
            axes[1].imshow(mask, cmap='RdYlGn')
            axes[1].set_title(f'Vegetation Mask ({method}, Coverage: {vegetation_ratio:.2f}%)', 
                              fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Right: Overlay display
            overlay = image_rgb.copy()
            highlight_color = np.array(index_colors[method])
            overlay[mask] = overlay[mask] * 0.7 + highlight_color * 0.3
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Vegetation Overlay', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save image
            mask_save_path = output_dir / f"vegetation_mask_{method.lower()}.png"
            plt.savefig(mask_save_path, dpi=300, bbox_inches='tight')
            print(f"  {method} visualization saved to: {mask_save_path}")
            plt.close()  # Close figure to release memory
            
        except Exception as e:
            print(f"  {method} processing failed: {e}")
            continue
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()

