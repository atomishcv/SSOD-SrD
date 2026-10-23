"""
DINOv3 Backbone 特征可视化模块
用于可视化多尺度特征图热力图
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


__all__ = [
    'visualize_dinov3_features',
    'visualize_dinov3_features_simple',
    'visualize_feature_on_image',
]


def visualize_dinov3_features(backbone_feats: List[torch.Tensor], 
                              images: torch.Tensor,
                              save_dir: str = './visualizations/dinov3_heatmaps',
                              step: int = 0,
                              epoch: int = 0,
                              max_samples: int = 2) -> None:
    """
    可视化 DINOv3 backbone 多尺度特征图热力图（完整版）
    
    Args:
        backbone_feats: DINOv3 backbone 输出的多尺度特征列表
                       [0]: [B, C, H1, W1] - 高分辨率
                       [1]: [B, C, H2, W2] - 中分辨率  
                       [2]: [B, C, H3, W3] - 低分辨率
        images: 原始输入图像 [B, 3, H, W]
        save_dir: 保存目录
        step: 当前训练步数
        epoch: 当前epoch
        max_samples: 最多可视化的样本数（避免生成过多图片）
    """
    if not isinstance(backbone_feats, (list, tuple)) or len(backbone_feats) == 0:
        print("⚠️ backbone_feats 不是列表/元组或为空")
        return
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 只在主进程中可视化
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except:
        pass
    
    batch_size = min(backbone_feats[0].shape[0], max_samples)
    
    for batch_idx in range(batch_size):
        # 创建多尺度特征可视化
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'DINOv3 Multi-Scale Feature Heatmaps (Epoch {epoch}, Step {step}, Sample {batch_idx})', 
                     fontsize=16, fontweight='bold')
        
        # 原始图像
        img = images[batch_idx].detach().cpu()
        if img.shape[0] == 3:  # [C, H, W]
            img = img.permute(1, 2, 0)  # [H, W, C]
        img = img.numpy()
        
        # 反归一化（假设使用 ImageNet 标准化）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # 显示原始图像
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 可视化三个尺度的特征
        for scale_idx, feat in enumerate(backbone_feats):
            # 提取单个样本的特征 [C, H, W]
            feat_map = feat[batch_idx].detach().cpu()
            
            # 转换 float16 到 float32
            if feat_map.dtype == torch.float16:
                feat_map = feat_map.float()
            
            C, H, W = feat_map.shape
            scale_name = f'{H}×{W}'
            
            # 方法1: 计算所有通道的平均激活（全局语义）
            avg_activation = feat_map.mean(dim=0).numpy()  # [H, W]
            
            # 方法2: 计算所有通道的最大激活（显著性区域）
            max_activation = feat_map.max(dim=0)[0].numpy()  # [H, W]
            
            # 归一化到 [0, 1]
            def normalize_map(x):
                x = x.astype(np.float32)
                x_min, x_max = x.min(), x.max()
                if x_max - x_min > 1e-6:
                    return (x - x_min) / (x_max - x_min)
                return x
            
            avg_activation = normalize_map(avg_activation)
            max_activation = normalize_map(max_activation)
            
            # 上采样到原始图像尺寸
            img_h, img_w = img.shape[:2]
            avg_heatmap = _smart_resize(avg_activation, (img_w, img_h))
            max_heatmap = _smart_resize(max_activation, (img_w, img_h))
            
            # 绘制三种热力图
            col_offset = scale_idx + 1
            
            # 平均激活热力图
            axes[0, col_offset].imshow(img)
            im1 = axes[0, col_offset].imshow(avg_heatmap, cmap='jet', alpha=0.6)
            axes[0, col_offset].set_title(f'{scale_name}\nAvg Activation', fontsize=10)
            axes[0, col_offset].axis('off')
            plt.colorbar(im1, ax=axes[0, col_offset], fraction=0.046, pad=0.04)
            
            # 最大激活热力图（显著性）
            axes[1, col_offset].imshow(img)
            im2 = axes[1, col_offset].imshow(max_heatmap, cmap='hot', alpha=0.6)
            axes[1, col_offset].set_title(f'{scale_name}\nMax Activation (Saliency)', fontsize=10)
            axes[1, col_offset].axis('off')
            plt.colorbar(im2, ax=axes[1, col_offset], fraction=0.046, pad=0.04)
        
        # 隐藏空白子图
        axes[1, 0].axis('off')
        
        # 保存图像
        save_file = save_path / f'epoch{epoch:03d}_step{step:05d}_sample{batch_idx}.png'
        plt.tight_layout()
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ DINOv3 特征热力图已保存: {save_file}")


def visualize_dinov3_features_simple(backbone_feats: List[torch.Tensor],
                                    images: torch.Tensor,
                                    save_dir: str = './visualizations/dinov3_heatmaps',
                                    step: int = 0,
                                    epoch: int = 0,
                                    sample_idx: int = 0,
                                    verbose: bool = True) -> None:
    """
    简化版 DINOv3 特征可视化（单样本，快速版本）
    
    Args:
        backbone_feats: DINOv3 backbone 输出的多尺度特征列表
                       [0]: [B, C, H1, W1] - 高分辨率
                       [1]: [B, C, H2, W2] - 中分辨率
                       [2]: [B, C, H3, W3] - 低分辨率
        images: 原始输入图像 [B, 3, H, W]
        save_dir: 保存目录
        step: 当前训练步数
        epoch: 当前epoch
        sample_idx: 要可视化的样本索引
        verbose: 是否打印详细调试信息
    """
    # 验证输入
    if verbose:
        print(f"\n🔍 [DINOv3可视化调试] Epoch {epoch}, Step {step}")
        print(f"   backbone_feats 类型: {type(backbone_feats)}")
        print(f"   backbone_feats 是否为 list: {isinstance(backbone_feats, list)}")
        print(f"   backbone_feats 是否为 tuple: {isinstance(backbone_feats, tuple)}")
    
    # 转换 tuple 为 list（如果需要）
    if isinstance(backbone_feats, tuple):
        if verbose:
            print(f"   ⚠️ 检测到 tuple，转换为 list")
        backbone_feats = list(backbone_feats)
    
    # 检查是否为列表或元组
    if not isinstance(backbone_feats, (list, tuple)):
        print(f"   ❌ backbone_feats 既不是 list 也不是 tuple，类型: {type(backbone_feats)}")
        return
    
    # 检查长度
    if len(backbone_feats) == 0:
        print(f"   ❌ backbone_feats 长度为 0")
        return
    
    if verbose:
        print(f"   ✅ backbone_feats 长度: {len(backbone_feats)}")
        
        # 打印每个元素的信息
        for idx, feat in enumerate(backbone_feats):
            if isinstance(feat, torch.Tensor):
                print(f"   [{idx}] Tensor shape: {list(feat.shape)}, device: {feat.device}, dtype: {feat.dtype}")
            else:
                print(f"   [{idx}] 不是 Tensor，类型: {type(feat)}")
                return
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 只在主进程中可视化
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except:
        pass
    
    # 检查 sample_idx 是否有效
    batch_size = backbone_feats[0].shape[0]
    if sample_idx >= batch_size:
        if verbose:
            print(f"⚠️ sample_idx={sample_idx} 超出 batch_size={batch_size}")
        sample_idx = 0
    
    # 提取单个样本的图像
    img = images[sample_idx].detach().cpu()
    if img.shape[0] == 3:  # [C, H, W] -> [H, W, C]
        img = img.permute(1, 2, 0).numpy()
    else:
        print(f"⚠️ 图像通道数异常: {img.shape}")
        return
    
    # 反归一化（ImageNet 标准）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # 创建图像网格
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'DINOv3 Multi-Scale Features (Epoch {epoch}, Step {step})', 
                 fontsize=14, fontweight='bold')
    
    # 显示原始图像
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 可视化三个尺度的特征
    img_h, img_w = img.shape[:2]
    
    for i, feat in enumerate(backbone_feats):
        # 提取单个样本的特征 [C, H, W]
        feat_map = feat[sample_idx].detach().cpu()
        C, H, W = feat_map.shape
        
        # ✅ 转换为 float32（OpenCV 不支持 float16）
        if feat_map.dtype == torch.float16:
            feat_map = feat_map.float()
        
        # 计算平均激活
        avg_activation = feat_map.mean(dim=0).numpy()  # [H, W]
        
        # ✅ 确保是 float32 类型
        if avg_activation.dtype == np.float16:
            avg_activation = avg_activation.astype(np.float32)
        
        if verbose:
            print(f"   [{i}] avg_activation shape: {avg_activation.shape}, dtype: {avg_activation.dtype}")
        
        # 归一化到 [0, 1]
        feat_min, feat_max = avg_activation.min(), avg_activation.max()
        if feat_max - feat_min > 1e-6:
            avg_activation = (avg_activation - feat_min) / (feat_max - feat_min)
        else:
            avg_activation = np.zeros_like(avg_activation, dtype=np.float32)
        
        # ✅ 智能上采样
        heatmap, kernel_size = _smart_resize(avg_activation, (img_w, img_h), return_kernel=True)
        upscale_ratio = img_h / H
        
        # 叠加显示
        axes[i+1].imshow(img)
        im = axes[i+1].imshow(heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        axes[i+1].set_title(f'Scale {H}×{W}\n(↑{upscale_ratio:.1f}x, blur={kernel_size})', 
                           fontsize=10)
        axes[i+1].axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)
        cbar.set_label('Activation', fontsize=8)
    
    # 保存
    save_file = save_path / f'epoch{epoch:03d}_step{step:05d}_sample{sample_idx}.png'
    plt.tight_layout()
    plt.savefig(save_file, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"✅ 特征热力图已保存: {save_file}")


def _smart_resize(feature_map: np.ndarray, 
                 target_size: tuple, 
                 return_kernel: bool = False):
    """
    智能多阶段上采样策略
    
    Args:
        feature_map: 特征图 [H, W]
        target_size: 目标尺寸 (W, H)
        return_kernel: 是否返回高斯核大小
        
    Returns:
        heatmap: 上采样后的热力图
        kernel_size: (可选) 高斯核大小
    """
    img_w, img_h = target_size
    H, W = feature_map.shape
    upscale_ratio = img_h / H
    
    try:
        if upscale_ratio > 4:
            # 大倍数上采样：分阶段进行
            # Stage 1: 使用 INTER_LINEAR 先放大到 2 倍
            heatmap = cv2.resize(feature_map.astype(np.float32), 
                                (W * 2, H * 2), 
                                interpolation=cv2.INTER_LINEAR)
            # Stage 2: 再使用 INTER_CUBIC 放大到目标尺寸
            heatmap = cv2.resize(heatmap, (img_w, img_h), 
                                interpolation=cv2.INTER_CUBIC)
        else:
            # 小倍数上采样：直接使用 INTER_CUBIC
            heatmap = cv2.resize(feature_map.astype(np.float32), 
                                (img_w, img_h), 
                                interpolation=cv2.INTER_CUBIC)
        
        # 对热力图应用高斯平滑（减少块状伪影）
        kernel_size = max(3, int(upscale_ratio // 2) * 2 + 1)  # 确保是奇数
        heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
        
    except Exception as e:
        print(f"   ⚠️ 上采样失败: {e}, 使用简单插值")
        # 降级方案：使用 scipy
        try:
            from scipy.ndimage import zoom
            zoom_factor = img_h / H
            heatmap = zoom(feature_map.astype(np.float32), zoom_factor, order=1)
            kernel_size = 3
        except:
            # 最终降级：直接使用 numpy
            heatmap = feature_map.astype(np.float32)
            kernel_size = 1
    
    # 再次归一化（平滑后可能改变范围）
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    
    if return_kernel:
        return heatmap, kernel_size
    return heatmap


def visualize_feature_on_image(feature_map: torch.Tensor,
                               image: torch.Tensor,
                               save_path: Optional[str] = None,
                               title: str = 'Feature Heatmap',
                               cmap: str = 'jet',
                               alpha: float = 0.6) -> np.ndarray:
    """
    将单个特征图可视化到图像上
    
    Args:
        feature_map: 特征图 [C, H, W] 或 [H, W]
        image: 输入图像 [3, H, W]
        save_path: 保存路径（可选）
        title: 图像标题
        cmap: 颜色映射
        alpha: 热力图透明度
        
    Returns:
        可视化结果 [H, W, 3]
    """
    # 转换到 CPU 和 numpy
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu()
        if feature_map.dtype == torch.float16:
            feature_map = feature_map.float()
        
        # 如果是多通道，计算平均
        if feature_map.dim() == 3:
            feature_map = feature_map.mean(dim=0)
        
        feature_map = feature_map.numpy()
    
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
    
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    # 归一化特征图
    feature_map = feature_map.astype(np.float32)
    feat_min, feat_max = feature_map.min(), feature_map.max()
    if feat_max - feat_min > 1e-6:
        feature_map = (feature_map - feat_min) / (feat_max - feat_min)
    
    # 上采样
    img_h, img_w = image.shape[:2]
    heatmap = _smart_resize(feature_map, (img_w, img_h))
    
    # 应用颜色映射
    heatmap_colored = plt.cm.get_cmap(cmap)(heatmap)[..., :3]
    
    # 叠加
    result = image * (1 - alpha) + heatmap_colored * alpha
    result = np.clip(result, 0, 1)
    
    # 保存
    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap=cmap)
        axes[1].set_title('Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(result)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return result

