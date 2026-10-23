"""
数据安全检查工具
用于检测和修复数据加载中的 NaN/Inf 问题
"""

import torch
import numpy as np


def check_and_fix_images(images, name="images", verbose=True):
    """
    检查并修复图像张量中的 NaN/Inf
    
    Args:
        images: 图像张量 [B, C, H, W]
        name: 张量名称，用于日志
        verbose: 是否打印详细信息
        
    Returns:
        cleaned_images: 清理后的图像张量
        has_issue: 是否检测到问题
    """
    has_nan = torch.isnan(images).any()
    has_inf = torch.isinf(images).any()
    
    if not has_nan and not has_inf:
        return images, False
    
    # 检测到问题
    if verbose:
        if has_nan:
            nan_ratio = torch.isnan(images).sum().item() / images.numel()
            print(f"⚠️  [{name}] NaN detected! Ratio: {nan_ratio:.2%}")
            
            # 按通道检查
            for c in range(images.shape[1]):
                channel_nan = torch.isnan(images[:, c]).any()
                if channel_nan:
                    print(f"   - Channel {c}: has NaN")
        
        if has_inf:
            inf_ratio = torch.isinf(images).sum().item() / images.numel()
            print(f"⚠️  [{name}] Inf detected! Ratio: {inf_ratio:.2%}")
    
    # 修复：用 0 替换 NaN，用边界值替换 Inf
    cleaned = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
    
    return cleaned, True


def check_targets(targets, name="targets", verbose=True):
    """
    检查目标字典中的 NaN/Inf
    
    Args:
        targets: 目标列表 [dict, dict, ...]
        name: 名称
        verbose: 是否打印详细信息
        
    Returns:
        has_issue: 是否检测到问题
    """
    has_issue = False
    
    for idx, target in enumerate(targets):
        if 'boxes' in target and isinstance(target['boxes'], torch.Tensor):
            if torch.isnan(target['boxes']).any():
                if verbose:
                    print(f"⚠️  [{name}][{idx}] NaN in boxes")
                has_issue = True
                # 修复
                target['boxes'] = torch.nan_to_num(target['boxes'], nan=0.5)
        
        if 'labels' in target and isinstance(target['labels'], torch.Tensor):
            if torch.isnan(target['labels']).any():
                if verbose:
                    print(f"⚠️  [{name}][{idx}] NaN in labels")
                has_issue = True
                # 修复：设置为背景类 (0)
                target['labels'] = torch.nan_to_num(target['labels'], nan=0.0).long()
    
    return has_issue


def diagnose_data_pipeline(images, targets=None, stage="unknown"):
    """
    诊断数据管道，打印详细信息
    
    Args:
        images: 图像张量
        targets: 目标列表（可选）
        stage: 当前阶段名称
    """
    print(f"\n🔍 [Data Diagnostic] Stage: {stage}")
    print(f"   Images shape: {images.shape}")
    print(f"   Images dtype: {images.dtype}")
    print(f"   Images device: {images.device}")
    print(f"   Images range: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"   Images mean: {images.mean().item():.4f}, std: {images.std().item():.4f}")
    
    has_nan = torch.isnan(images).any()
    has_inf = torch.isinf(images).any()
    
    if has_nan:
        nan_ratio = torch.isnan(images).sum().item() / images.numel()
        print(f"   ❌ NaN detected: {nan_ratio:.2%}")
    else:
        print(f"   ✅ No NaN")
    
    if has_inf:
        inf_ratio = torch.isinf(images).sum().item() / images.numel()
        print(f"   ❌ Inf detected: {inf_ratio:.2%}")
    else:
        print(f"   ✅ No Inf")
    
    if targets is not None:
        print(f"   Targets count: {len(targets)}")
        for idx, target in enumerate(targets):
            if 'boxes' in target:
                print(f"   Target[{idx}] boxes: {target['boxes'].shape if isinstance(target['boxes'], torch.Tensor) else 'N/A'}")
                if isinstance(target['boxes'], torch.Tensor) and torch.isnan(target['boxes']).any():
                    print(f"      ❌ Has NaN in boxes")

