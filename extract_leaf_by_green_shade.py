#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于绿色深浅区分不同种类植物的叶片提取脚本

通过HSV颜色空间中的亮度(Value)通道来区分深绿色和浅绿色植株，
从而分离不同种类的植物叶片。

Author: DEIMv2
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import argparse
from pathlib import Path


def load_image(image_path):
    """加载图片"""
    print(f"📖 Loading image: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"   Image size: {image_bgr.shape[1]} x {image_bgr.shape[0]} pixels")
    
    return image_bgr, image_rgb


def convert_to_hsv(image_bgr):
    """转换到HSV颜色空间"""
    print("\n🎨 Converting to HSV color space...")
    
    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    h, s, v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
    
    print(f"   Hue range: {h.min()} - {h.max()}")
    print(f"   Saturation range: {s.min()} - {s.max()}")
    print(f"   Value range: {v.min()} - {v.max()}")
    
    return hsv_image, h, s, v


def extract_green_regions(hsv_image, h_range=(25, 95), s_min=30, v_min=30):
    """
    提取绿色区域
    
    Args:
        hsv_image: HSV图像
        h_range: 色调范围 (H_min, H_max)，绿色通常在35-85之间
        s_min: 最小饱和度
        v_min: 最小亮度
    """
    print(f"\n🌿 Extracting green regions...")
    print(f"   Hue range: {h_range[0]} - {h_range[1]}")
    print(f"   Saturation min: {s_min}")
    print(f"   Value min: {v_min}")
    
    # 定义绿色范围
    lower_green = np.array([h_range[0], s_min, v_min])
    upper_green = np.array([h_range[1], 255, 255])
    
    # 创建绿色掩码
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # 形态学操作清理噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    green_pixels = np.sum(green_mask > 0)
    total_pixels = green_mask.shape[0] * green_mask.shape[1]
    print(f"   Green region: {green_pixels} pixels ({100*green_pixels/total_pixels:.1f}% of image)")
    
    return green_mask


def separate_by_green_shade(hsv_image, green_mask, v_threshold=None, method='adaptive'):
    """
    基于绿色深浅分离不同种类的植物
    
    Args:
        hsv_image: HSV图像
        green_mask: 绿色区域掩码
        v_threshold: 亮度阈值（None则自动计算）
        method: 'fixed' 或 'adaptive' 或 'kmeans'
    """
    print(f"\n🔬 Separating plants by green shade (method: {method})...")
    
    v_channel = hsv_image[:,:,2]
    h_channel = hsv_image[:,:,0]
    s_channel = hsv_image[:,:,1]
    
    # 只考虑绿色区域内的像素
    green_v_values = v_channel[green_mask > 0]
    green_h_values = h_channel[green_mask > 0]
    green_s_values = s_channel[green_mask > 0]
    
    if len(green_v_values) == 0:
        print("   No green pixels found!")
        return None, None, None, None
    
    print(f"   Green region Value stats: min={green_v_values.min()}, max={green_v_values.max()}, mean={green_v_values.mean():.1f}")
    print(f"   Green region Hue stats: min={green_h_values.min()}, max={green_h_values.max()}, mean={green_h_values.mean():.1f}")
    print(f"   Green region Saturation stats: min={green_s_values.min()}, max={green_s_values.max()}, mean={green_s_values.mean():.1f}")
    
    if method == 'fixed':
        # 固定阈值方法
        if v_threshold is None:
            v_threshold = np.median(green_v_values)
        print(f"   Using fixed V threshold: {v_threshold}")
        
        deep_green_mask = (green_mask > 0) & (v_channel < v_threshold)
        light_green_mask = (green_mask > 0) & (v_channel >= v_threshold)
        
    elif method == 'adaptive':
        # 自适应方法：使用Otsu阈值
        green_v_only = np.zeros_like(v_channel)
        green_v_only[green_mask > 0] = v_channel[green_mask > 0]
        
        # 对绿色区域应用Otsu阈值
        v_threshold, _ = cv2.threshold(green_v_only[green_mask > 0].astype(np.uint8), 
                                        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"   Otsu V threshold: {v_threshold}")
        
        deep_green_mask = (green_mask > 0) & (v_channel < v_threshold)
        light_green_mask = (green_mask > 0) & (v_channel >= v_threshold)
        
    elif method == 'kmeans':
        # K-means聚类方法
        # 使用HSV三个通道进行聚类
        green_pixels_hsv = np.column_stack([
            h_channel[green_mask > 0],
            s_channel[green_mask > 0],
            v_channel[green_mask > 0]
        ]).astype(np.float32)
        
        # 归一化
        green_pixels_hsv[:, 0] = green_pixels_hsv[:, 0] / 180.0  # H: 0-180
        green_pixels_hsv[:, 1] = green_pixels_hsv[:, 1] / 255.0  # S: 0-255
        green_pixels_hsv[:, 2] = green_pixels_hsv[:, 2] / 255.0  # V: 0-255
        
        # K-means聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(green_pixels_hsv, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 根据亮度中心确定深绿和浅绿
        if centers[0, 2] < centers[1, 2]:  # 比较V值
            deep_label, light_label = 0, 1
        else:
            deep_label, light_label = 1, 0
        
        print(f"   K-means centers (H,S,V normalized):")
        print(f"      Deep green: H={centers[deep_label,0]*180:.1f}, S={centers[deep_label,1]*255:.1f}, V={centers[deep_label,2]*255:.1f}")
        print(f"      Light green: H={centers[light_label,0]*180:.1f}, S={centers[light_label,1]*255:.1f}, V={centers[light_label,2]*255:.1f}")
        
        # 创建掩码
        deep_green_mask = np.zeros_like(green_mask, dtype=bool)
        light_green_mask = np.zeros_like(green_mask, dtype=bool)
        
        green_coords = np.where(green_mask > 0)
        for i, (y, x) in enumerate(zip(green_coords[0], green_coords[1])):
            if labels[i] == deep_label:
                deep_green_mask[y, x] = True
            else:
                light_green_mask[y, x] = True
        
        v_threshold = (centers[deep_label, 2] + centers[light_label, 2]) / 2 * 255
    
    elif method == 'multi_channel':
        # 多通道综合方法：结合H、S、V进行分离
        print("   Using multi-channel separation...")
        
        # 计算综合指标：深绿色通常H值较高（更偏蓝绿）、V值较低
        # 浅绿色通常H值较低（更偏黄绿）、V值较高
        
        # 归一化
        h_norm = h_channel.astype(float) / 180.0
        s_norm = s_channel.astype(float) / 255.0
        v_norm = v_channel.astype(float) / 255.0
        
        # 综合评分：深绿色得分高
        # 考虑：高饱和度 + 低亮度 = 深绿色
        deep_score = s_norm * (1 - v_norm)
        
        # 计算阈值
        score_threshold = np.median(deep_score[green_mask > 0])
        print(f"   Deep green score threshold: {score_threshold:.3f}")
        
        deep_green_mask = (green_mask > 0) & (deep_score >= score_threshold)
        light_green_mask = (green_mask > 0) & (deep_score < score_threshold)
        
        v_threshold = score_threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 转换为uint8
    deep_green_mask = deep_green_mask.astype(np.uint8) * 255
    light_green_mask = light_green_mask.astype(np.uint8) * 255
    
    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    deep_green_mask = cv2.morphologyEx(deep_green_mask, cv2.MORPH_OPEN, kernel)
    deep_green_mask = cv2.morphologyEx(deep_green_mask, cv2.MORPH_CLOSE, kernel)
    light_green_mask = cv2.morphologyEx(light_green_mask, cv2.MORPH_OPEN, kernel)
    light_green_mask = cv2.morphologyEx(light_green_mask, cv2.MORPH_CLOSE, kernel)
    
    deep_pixels = np.sum(deep_green_mask > 0)
    light_pixels = np.sum(light_green_mask > 0)
    print(f"   Deep green: {deep_pixels} pixels")
    print(f"   Light green: {light_pixels} pixels")
    
    return deep_green_mask, light_green_mask, v_threshold, green_v_values


def find_contours_and_features(mask, min_area=50, max_area_ratio=0.3, image_shape=None):
    """查找轮廓并提取形态特征"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = float('inf')
    if image_shape is not None:
        max_area = image_shape[0] * image_shape[1] * max_area_ratio
    
    features_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            perimeter = cv2.arcLength(contour, True)
            
            # 最小外接矩形
            if len(contour) >= 5:
                rect = cv2.minAreaRect(contour)
                width = max(rect[1])
                height = min(rect[1])
                aspect_ratio = width / (height + 1e-6)
            else:
                aspect_ratio = 1.0
            
            # 圆形度
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0
            
            # 凸包
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            
            # 质心
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0
            
            features_list.append({
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'solidity': solidity,
                'centroid': (cx, cy),
                'hull': hull
            })
    
    return features_list


def visualize_hsv_analysis(image_rgb, hsv_image, h, s, v, output_dir):
    """可视化HSV分析"""
    print("\n🎨 Visualizing HSV analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # H通道
    im_h = axes[0, 1].imshow(h, cmap='hsv', vmin=0, vmax=180)
    axes[0, 1].set_title('Hue Channel (0-180)', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im_h, ax=axes[0, 1], fraction=0.046)
    
    # S通道
    im_s = axes[0, 2].imshow(s, cmap='Greens', vmin=0, vmax=255)
    axes[0, 2].set_title('Saturation Channel (0-255)', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im_s, ax=axes[0, 2], fraction=0.046)
    
    # V通道
    im_v = axes[1, 0].imshow(v, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title('Value Channel (0-255)', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im_v, ax=axes[1, 0], fraction=0.046)
    
    # H通道直方图
    axes[1, 1].hist(h.flatten(), bins=90, range=(0, 180), color='orange', alpha=0.7)
    axes[1, 1].axvline(35, color='green', linestyle='--', label='Green range start')
    axes[1, 1].axvline(85, color='green', linestyle='--', label='Green range end')
    axes[1, 1].set_xlabel('Hue')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Hue Distribution', fontsize=12)
    axes[1, 1].legend()
    
    # V通道直方图
    axes[1, 2].hist(v.flatten(), bins=50, range=(0, 255), color='gray', alpha=0.7)
    axes[1, 2].set_xlabel('Value (Brightness)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Value Distribution', fontsize=12)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'hsv_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_path}")


def visualize_green_separation(image_rgb, green_mask, deep_mask, light_mask, 
                                v_threshold, green_v_values, output_dir):
    """可视化绿色分离结果"""
    print("\n🎨 Visualizing green separation...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原图
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # 绿色区域
    axes[0, 1].imshow(green_mask, cmap='Greens')
    axes[0, 1].set_title('Green Region Mask', fontsize=12)
    axes[0, 1].axis('off')
    
    # 绿色区域V值分布
    if green_v_values is not None and len(green_v_values) > 0:
        axes[0, 2].hist(green_v_values, bins=50, color='green', alpha=0.7, edgecolor='white')
        if v_threshold is not None:
            axes[0, 2].axvline(v_threshold, color='red', linestyle='--', linewidth=2, 
                              label=f'Threshold: {v_threshold:.1f}')
        axes[0, 2].set_xlabel('Value (Brightness)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Green Region Value Distribution', fontsize=12)
        axes[0, 2].legend()
    
    # 深绿色区域
    deep_overlay = image_rgb.copy()
    deep_overlay[deep_mask > 0] = [0, 100, 0]  # 深绿色
    deep_blended = cv2.addWeighted(image_rgb, 0.5, deep_overlay, 0.5, 0)
    axes[1, 0].imshow(deep_blended)
    deep_count = np.sum(deep_mask > 0)
    axes[1, 0].set_title(f'Deep Green ({deep_count} pixels)', fontsize=12)
    axes[1, 0].axis('off')
    
    # 浅绿色区域
    light_overlay = image_rgb.copy()
    light_overlay[light_mask > 0] = [144, 238, 144]  # 浅绿色
    light_blended = cv2.addWeighted(image_rgb, 0.5, light_overlay, 0.5, 0)
    axes[1, 1].imshow(light_blended)
    light_count = np.sum(light_mask > 0)
    axes[1, 1].set_title(f'Light Green ({light_count} pixels)', fontsize=12)
    axes[1, 1].axis('off')
    
    # 综合显示：深绿色=蓝色，浅绿色=黄色
    combined = image_rgb.copy()
    combined[deep_mask > 0] = [0, 0, 200]  # 蓝色表示深绿
    combined[light_mask > 0] = [255, 255, 0]  # 黄色表示浅绿
    combined_blended = cv2.addWeighted(image_rgb, 0.4, combined, 0.6, 0)
    axes[1, 2].imshow(combined_blended)
    axes[1, 2].set_title('Combined (Blue=Deep, Yellow=Light)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'green_separation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_path}")


def visualize_plant_contours(image_rgb, deep_features, light_features, output_dir):
    """可视化不同类型植物的轮廓"""
    print("\n🎨 Visualizing plant contours...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # 深绿色植物轮廓
    img_deep = image_rgb.copy()
    for f in deep_features:
        cv2.drawContours(img_deep, [f['contour']], -1, (0, 0, 255), 2)  # 蓝色
        cx, cy = f['centroid']
        cv2.circle(img_deep, (cx, cy), 4, (255, 0, 0), -1)
    axes[0, 0].imshow(img_deep)
    axes[0, 0].set_title(f'Deep Green Plants (n={len(deep_features)})\n(Blue contours)', fontsize=12)
    axes[0, 0].axis('off')
    
    # 浅绿色植物轮廓
    img_light = image_rgb.copy()
    for f in light_features:
        cv2.drawContours(img_light, [f['contour']], -1, (255, 255, 0), 2)  # 黄色
        cx, cy = f['centroid']
        cv2.circle(img_light, (cx, cy), 4, (255, 0, 0), -1)
    axes[0, 1].imshow(img_light)
    axes[0, 1].set_title(f'Light Green Plants (n={len(light_features)})\n(Yellow contours)', fontsize=12)
    axes[0, 1].axis('off')
    
    # 综合显示
    img_combined = image_rgb.copy()
    for f in deep_features:
        cv2.drawContours(img_combined, [f['contour']], -1, (0, 0, 255), 2)
    for f in light_features:
        cv2.drawContours(img_combined, [f['contour']], -1, (255, 255, 0), 2)
    axes[1, 0].imshow(img_combined)
    axes[1, 0].set_title('All Plants Combined\n(Blue=Deep, Yellow=Light)', fontsize=12)
    axes[1, 0].axis('off')
    
    # 填充显示
    img_filled = image_rgb.copy()
    for f in deep_features:
        cv2.drawContours(img_filled, [f['contour']], -1, (0, 0, 200), -1)  # 填充蓝色
    for f in light_features:
        cv2.drawContours(img_filled, [f['contour']], -1, (255, 255, 100), -1)  # 填充黄色
    img_filled_blended = cv2.addWeighted(image_rgb, 0.5, img_filled, 0.5, 0)
    axes[1, 1].imshow(img_filled_blended)
    axes[1, 1].set_title('Filled Regions\n(Blue=Deep, Yellow=Light)', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'plant_contours.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_path}")


def visualize_morphology_comparison(deep_features, light_features, output_dir):
    """可视化两类植物的形态特征对比"""
    print("\n📊 Visualizing morphology comparison...")
    
    if len(deep_features) == 0 and len(light_features) == 0:
        print("   No features to compare!")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 提取特征
    deep_areas = [f['area'] for f in deep_features] if deep_features else [0]
    light_areas = [f['area'] for f in light_features] if light_features else [0]
    
    deep_circs = [f['circularity'] for f in deep_features] if deep_features else [0]
    light_circs = [f['circularity'] for f in light_features] if light_features else [0]
    
    deep_ars = [f['aspect_ratio'] for f in deep_features] if deep_features else [0]
    light_ars = [f['aspect_ratio'] for f in light_features] if light_features else [0]
    
    deep_sols = [f['solidity'] for f in deep_features] if deep_features else [0]
    light_sols = [f['solidity'] for f in light_features] if light_features else [0]
    
    # 1. 面积对比
    axes[0, 0].boxplot([deep_areas, light_areas], labels=['Deep Green', 'Light Green'])
    axes[0, 0].set_ylabel('Area (px²)')
    axes[0, 0].set_title('Area Comparison', fontsize=12)
    
    # 2. 圆形度对比
    axes[0, 1].boxplot([deep_circs, light_circs], labels=['Deep Green', 'Light Green'])
    axes[0, 1].set_ylabel('Circularity')
    axes[0, 1].set_title('Circularity Comparison', fontsize=12)
    
    # 3. 长宽比对比
    axes[0, 2].boxplot([deep_ars, light_ars], labels=['Deep Green', 'Light Green'])
    axes[0, 2].set_ylabel('Aspect Ratio')
    axes[0, 2].set_title('Aspect Ratio Comparison', fontsize=12)
    
    # 4. 凸度对比
    axes[1, 0].boxplot([deep_sols, light_sols], labels=['Deep Green', 'Light Green'])
    axes[1, 0].set_ylabel('Solidity')
    axes[1, 0].set_title('Solidity Comparison', fontsize=12)
    
    # 5. 面积分布直方图
    if len(deep_features) > 0:
        axes[1, 1].hist(deep_areas, bins=20, alpha=0.6, label='Deep Green', color='blue')
    if len(light_features) > 0:
        axes[1, 1].hist(light_areas, bins=20, alpha=0.6, label='Light Green', color='yellow', edgecolor='gray')
    axes[1, 1].set_xlabel('Area (px²)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Area Distribution', fontsize=12)
    axes[1, 1].legend()
    
    # 6. 统计表格
    axes[1, 2].axis('off')
    
    stats_data = [
        ['Metric', 'Deep Green', 'Light Green'],
        ['Count', f'{len(deep_features)}', f'{len(light_features)}'],
        ['Area (mean)', f'{np.mean(deep_areas):.1f}', f'{np.mean(light_areas):.1f}'],
        ['Area (std)', f'{np.std(deep_areas):.1f}', f'{np.std(light_areas):.1f}'],
        ['Circularity', f'{np.mean(deep_circs):.3f}', f'{np.mean(light_circs):.3f}'],
        ['Aspect Ratio', f'{np.mean(deep_ars):.2f}', f'{np.mean(light_ars):.2f}'],
        ['Solidity', f'{np.mean(deep_sols):.3f}', f'{np.mean(light_sols):.3f}'],
    ]
    
    table = axes[1, 2].table(cellText=stats_data, loc='center', cellLoc='center',
                             colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    axes[1, 2].set_title('Statistical Comparison', fontsize=12, pad=20)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'morphology_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_path}")


def create_summary_figure(image_rgb, deep_mask, light_mask, deep_features, light_features, output_dir):
    """创建综合总结图"""
    print("\n📋 Creating summary figure...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 原图 (左上)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image_rgb)
    ax1.set_title('Original Image', fontsize=14)
    ax1.axis('off')
    
    # 2. 分离结果 (中上)
    ax2 = fig.add_subplot(2, 3, 2)
    combined = image_rgb.copy()
    combined[deep_mask > 0] = [0, 0, 200]
    combined[light_mask > 0] = [255, 255, 0]
    combined_blended = cv2.addWeighted(image_rgb, 0.4, combined, 0.6, 0)
    ax2.imshow(combined_blended)
    ax2.set_title('Separated by Green Shade\n(Blue=Deep, Yellow=Light)', fontsize=14)
    ax2.axis('off')
    
    # 3. 轮廓检测 (右上)
    ax3 = fig.add_subplot(2, 3, 3)
    img_contours = image_rgb.copy()
    for f in deep_features:
        cv2.drawContours(img_contours, [f['contour']], -1, (0, 0, 255), 2)
    for f in light_features:
        cv2.drawContours(img_contours, [f['contour']], -1, (255, 255, 0), 2)
    ax3.imshow(img_contours)
    ax3.set_title(f'Detected Plants\nDeep: {len(deep_features)}, Light: {len(light_features)}', fontsize=14)
    ax3.axis('off')
    
    # 4. 深绿色植物 (左下)
    ax4 = fig.add_subplot(2, 3, 4)
    deep_only = np.zeros_like(image_rgb)
    deep_only[deep_mask > 0] = image_rgb[deep_mask > 0]
    ax4.imshow(deep_only)
    ax4.set_title(f'Deep Green Plants Only\n({len(deep_features)} detected)', fontsize=14)
    ax4.axis('off')
    
    # 5. 浅绿色植物 (中下)
    ax5 = fig.add_subplot(2, 3, 5)
    light_only = np.zeros_like(image_rgb)
    light_only[light_mask > 0] = image_rgb[light_mask > 0]
    ax5.imshow(light_only)
    ax5.set_title(f'Light Green Plants Only\n({len(light_features)} detected)', fontsize=14)
    ax5.axis('off')
    
    # 6. 统计信息 (右下)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # 统计信息
    deep_areas = [f['area'] for f in deep_features] if deep_features else [0]
    light_areas = [f['area'] for f in light_features] if light_features else [0]
    
    info_text = f"""
    === PLANT SEPARATION RESULTS ===
    
    DEEP GREEN PLANTS:
      • Count: {len(deep_features)}
      • Total Area: {np.sum(deep_areas):.0f} px²
      • Mean Area: {np.mean(deep_areas):.1f} px²
      • Area Range: {np.min(deep_areas):.0f} - {np.max(deep_areas):.0f} px²
    
    LIGHT GREEN PLANTS:
      • Count: {len(light_features)}
      • Total Area: {np.sum(light_areas):.0f} px²
      • Mean Area: {np.mean(light_areas):.1f} px²
      • Area Range: {np.min(light_areas):.0f} - {np.max(light_areas):.0f} px²
    
    RATIO:
      • Deep/Light Count: {len(deep_features)/(len(light_features)+1e-6):.2f}
      • Deep/Light Area: {np.sum(deep_areas)/(np.sum(light_areas)+1e-6):.2f}
    """
    
    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax6.set_title('Statistics', fontsize=14)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'green_shade_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Separate plants by green shade')
    parser.add_argument('--image', type=str, 
                       default='DJI_dataset/images/coco_dataset_4_DJI_20250626115005_0131_patch_2766_3688.jpg',
                       help='Input image path')
    parser.add_argument('--output-dir', type=str, default='vegetation_analysis/green_shade',
                       help='Output directory')
    parser.add_argument('--method', type=str, default='adaptive', 
                       choices=['fixed', 'adaptive', 'kmeans', 'multi_channel'],
                       help='Separation method')
    parser.add_argument('--h-min', type=int, default=25,
                       help='Minimum hue for green (default 25)')
    parser.add_argument('--h-max', type=int, default=95,
                       help='Maximum hue for green (default 95)')
    parser.add_argument('--s-min', type=int, default=30,
                       help='Minimum saturation (default 30)')
    parser.add_argument('--v-min', type=int, default=30,
                       help='Minimum value (default 30)')
    parser.add_argument('--v-threshold', type=int, default=None,
                       help='Fixed value threshold (only for fixed method)')
    parser.add_argument('--min-area', type=int, default=50,
                       help='Minimum contour area (default 50)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🌿 Plant Separation by Green Shade")
    print("="*80)
    print(f"Image: {args.image}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    print(f"Hue range: {args.h_min} - {args.h_max}")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载图像
    image_bgr, image_rgb = load_image(args.image)
    
    # 2. 转换到HSV
    hsv_image, h, s, v = convert_to_hsv(image_bgr)
    
    # 3. 可视化HSV分析
    visualize_hsv_analysis(image_rgb, hsv_image, h, s, v, args.output_dir)
    
    # 4. 提取绿色区域
    green_mask = extract_green_regions(hsv_image, 
                                        h_range=(args.h_min, args.h_max),
                                        s_min=args.s_min, 
                                        v_min=args.v_min)
    
    # 5. 基于绿色深浅分离
    deep_mask, light_mask, v_threshold, green_v_values = separate_by_green_shade(
        hsv_image, green_mask, 
        v_threshold=args.v_threshold,
        method=args.method
    )
    
    if deep_mask is None:
        print("\n❌ Failed to separate plants. Exiting.")
        return
    
    # 6. 可视化分离结果
    visualize_green_separation(image_rgb, green_mask, deep_mask, light_mask,
                               v_threshold, green_v_values, args.output_dir)
    
    # 7. 查找轮廓和提取特征
    print("\n🔍 Finding plant contours...")
    deep_features = find_contours_and_features(deep_mask, min_area=args.min_area,
                                                image_shape=image_rgb.shape[:2])
    light_features = find_contours_and_features(light_mask, min_area=args.min_area,
                                                 image_shape=image_rgb.shape[:2])
    print(f"   Deep green plants: {len(deep_features)}")
    print(f"   Light green plants: {len(light_features)}")
    
    # 8. 可视化轮廓
    visualize_plant_contours(image_rgb, deep_features, light_features, args.output_dir)
    
    # 9. 可视化形态特征对比
    visualize_morphology_comparison(deep_features, light_features, args.output_dir)
    
    # 10. 创建总结图
    create_summary_figure(image_rgb, deep_mask, light_mask, deep_features, light_features, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ Plant separation complete!")
    print(f"   Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

