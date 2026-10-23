"""
单张图片测试脚本

使用训练好的模型对单张图片进行目标检测推理

使用方式：
    python test_single_image.py \
        --checkpoint outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth \
        --image path/to/image.jpg \
        --config configs/deimv2_ssod/deimv2_dinov3_x_coco.yml \
        --output visualizations/test_results/
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
import random
import glob
matplotlib.use('Agg')  # 使用非交互式后端

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from engine.core import YAMLConfig

# COCO 类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 用于可视化的颜色（BGR格式）
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128)
]


def load_model(checkpoint_path, config_path, device='cuda'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点文件路径
        config_path: 配置文件路径
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        model: 加载好的模型
        cfg: 配置对象
    """
    print(f"\n{'='*80}")
    print("📦 加载模型...")
    print(f"{'='*80}")
    print(f"检查点: {checkpoint_path}")
    print(f"配置文件: {config_path}")
    print(f"设备: {device}\n")
    
    # 加载配置
    cfg = YAMLConfig(config_path, resume=None)
    
    # 创建模型
    model = cfg.model
    model = model.to(device)
    model.eval()
    
    # 加载权重
    print("加载检查点权重...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的检查点格式
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'ema' in checkpoint:
        # 如果有 EMA 权重，优先使用
        ema_state = checkpoint['ema']
        if isinstance(ema_state, dict) and 'module' in ema_state:
            state_dict = ema_state['module']
        else:
            state_dict = ema_state
    else:
        state_dict = checkpoint
    
    # 移除 'module.' 前缀（如果有）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # 重要：强制使用动态位置编码，避免固定尺寸位置编码导致的尺寸不匹配
    # 将 encoder 和 decoder 的 eval_spatial_size 设置为 None，使其在推理时动态生成锚点
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'eval_spatial_size'):
        model.encoder.eval_spatial_size = None
        print("✅ 已设置 encoder 动态位置编码（eval_spatial_size=None）")
    
    # 🔥 关键修复：decoder 也需要设置 eval_spatial_size=None，否则会使用预缓存的锚点导致维度不匹配
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'eval_spatial_size'):
        model.decoder.eval_spatial_size = None
        print("✅ 已设置 decoder 动态锚点生成（eval_spatial_size=None）")
    
    print(f"✅ 模型加载完成")
    if 'epoch' in checkpoint:
        print(f"   - Epoch: {checkpoint['epoch']}")
    if 'ap' in checkpoint:
        print(f"   - AP: {checkpoint['ap']:.4f}")
    print()
    
    return model, cfg


class FeatureExtractor:
    """
    特征提取器 - 使用 hooks 捕获中间层特征
    """
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册 forward hooks 来捕获特征"""
        
        # Hook for Backbone output
        def hook_backbone(module, input, output):
            # output is tuple of (c2, c3, c4)
            self.features['backbone'] = [f.detach() for f in output]
        
        # Hook for Encoder output  
        def hook_encoder(module, input, output):
            # output is tuple of encoder features
            self.features['encoder'] = [f.detach() for f in output]
        
        # Hook for Decoder output (before final prediction)
        def hook_decoder(module, input, output):
            # Capture decoder features if available
            if hasattr(module, 'dec_output'):
                self.features['decoder'] = module.dec_output.detach()
        
        # 注册 hooks
        if hasattr(self.model, 'backbone'):
            h = self.model.backbone.register_forward_hook(hook_backbone)
            self.hooks.append(h)
            
        if hasattr(self.model, 'encoder'):
            h = self.model.encoder.register_forward_hook(hook_encoder)
            self.hooks.append(h)
            
        if hasattr(self.model, 'decoder'):
            h = self.model.decoder.register_forward_hook(hook_decoder)
            self.hooks.append(h)
    
    def remove_hooks(self):
        """移除所有 hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_features(self):
        """获取捕获的特征"""
        return self.features


def visualize_feature_maps(features, orig_image, output_dir, image_name):
    """
    可视化特征图
    
    Args:
        features: 特征字典 {'backbone': [...], 'encoder': [...], 'decoder': ...}
        orig_image: 原始图片（numpy数组，BGR格式）
        output_dir: 输出目录
        image_name: 图片名称
    """
    print("\n🎨 可视化中间层特征...")
    
    feature_dir = output_dir / 'features' / image_name
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换原始图片为 RGB
    orig_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig_image.shape[:2]
    
    # 1. 可视化 Backbone 特征
    if 'backbone' in features:
        print("   - Backbone 特征 (C2, C3, C4)...")
        backbone_feats = features['backbone']
        
        for i, feat in enumerate(backbone_feats):
            if feat is None:
                continue
            
            # feat shape: [B, C, H, W]
            feat_np = feat[0].cpu().numpy()  # [C, H, W]
            
            # 取所有通道的平均值作为热力图
            heatmap = feat_np.mean(axis=0)  # [H, W]
            
            # 归一化到 [0, 255]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图片
            axes[0].imshow(orig_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 热力图
            im = axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title(f'Backbone C{i+2} Heatmap\nShape: {feat_np.shape}')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # 叠加到原图
            heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8), 
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap_colored, 0.4, 0)
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = feature_dir / f'backbone_c{i+2}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      ✅ 保存: {save_path}")
    
    # 2. 可视化 Encoder 特征
    if 'encoder' in features:
        print("   - Encoder 特征...")
        encoder_feats = features['encoder']
        
        for i, feat in enumerate(encoder_feats):
            if feat is None:
                continue
            
            # feat shape: [B, C, H, W]
            feat_np = feat[0].cpu().numpy()  # [C, H, W]
            
            # 取所有通道的平均值
            heatmap = feat_np.mean(axis=0)  # [H, W]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(orig_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            im = axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title(f'Encoder Level {i} Heatmap\nShape: {feat_np.shape}')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap_colored, 0.4, 0)
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = feature_dir / f'encoder_level{i}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      ✅ 保存: {save_path}")
    
    # 3. 可视化多个通道的特征图（Backbone C4 的前 16 个通道）
    if 'backbone' in features and len(features['backbone']) > 0:
        print("   - Backbone 多通道特征...")
        c4_feat = features['backbone'][-1]  # C4 特征
        if c4_feat is not None:
            feat_np = c4_feat[0].cpu().numpy()  # [C, H, W]
            num_channels = min(16, feat_np.shape[0])
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.flatten()
            
            for idx in range(num_channels):
                channel_feat = feat_np[idx]
                channel_feat = (channel_feat - channel_feat.min()) / \
                              (channel_feat.max() - channel_feat.min() + 1e-8)
                
                axes[idx].imshow(channel_feat, cmap='viridis')
                axes[idx].set_title(f'Ch {idx}')
                axes[idx].axis('off')
            
            # 隐藏多余的子图
            for idx in range(num_channels, 16):
                axes[idx].axis('off')
            
            plt.suptitle('Backbone C4 - First 16 Channels', fontsize=16)
            plt.tight_layout()
            save_path = feature_dir / 'backbone_c4_channels.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      ✅ 保存: {save_path}")
    
    print(f"✅ 特征可视化完成！保存到: {feature_dir}")
    return feature_dir


def split_image_into_patches(image, patch_size=800, overlap=0.2):
    """
    将大图划分成多个 patch（滑动窗口）
    
    Args:
        image: 原始图片（numpy数组，BGR格式）
        patch_size: patch 尺寸（默认 800）
        overlap: 重叠比例（默认 0.2，即 20% 重叠）
    
    Returns:
        patches: patch 列表，每个元素为 (patch_image, patch_info)
        patch_info 包含: (x_start, y_start, x_end, y_end) 在原图中的坐标
    """
    h, w = image.shape[:2]
    stride = int(patch_size * (1 - overlap))  # 步长
    
    patches = []
    patch_infos = []
    
    # 滑动窗口划分
    y = 0
    while y < h:
        x = 0
        while x < w:
            # 计算 patch 的实际范围
            x_end = min(x + patch_size, w)
            y_end = min(y + patch_size, h)
            
            # 提取 patch
            patch = image[y:y_end, x:x_end]
            
            # 如果 patch 尺寸不足，进行 padding
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded_patch
            
            patches.append(patch)
            patch_infos.append((x, y, x_end, y_end))
            
            # 移动到下一个位置
            if x_end >= w:
                break
            x += stride
        
        if y_end >= h:
            break
        y += stride
    
    return patches, patch_infos


def preprocess_patch(patch_image, input_size=800):
    """
    预处理单个 patch
    
    Args:
        patch_image: patch 图片（numpy数组，BGR格式，已为 input_size x input_size）
        input_size: 输入尺寸（默认 800）
    
    Returns:
        image_tensor: 预处理后的图片张量 (1, 3, input_size, input_size)
    """
    # 转换为 RGB
    image_rgb = cv2.cvtColor(patch_image, cv2.COLOR_BGR2RGB)
    
    # 转换为 PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # 转换为 tensor 并归一化
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(pil_image).unsqueeze(0)  # (1, 3, input_size, input_size)
    
    return image_tensor


def preprocess_image(image_path, input_size=800, use_patch_mode=False, patch_size=800, overlap=0.2):
    """
    预处理图片 - 支持单张图片或 patch 模式
    
    Args:
        image_path: 图片路径
        input_size: 输入尺寸（默认 800）
        use_patch_mode: 是否使用 patch 模式（用于 8K 大图）
        patch_size: patch 尺寸（默认 800）
        overlap: 重叠比例（默认 0.2）
    
    Returns:
        如果 use_patch_mode=False:
            image_tensor: 预处理后的图片张量 (1, 3, input_size, input_size)
            orig_image: 原始图片
            scale: 缩放比例
            pad_info: padding 信息
        如果 use_patch_mode=True:
            patches: patch 列表
            patch_infos: patch 信息列表
            orig_image: 原始图片
    """
    # 读取图片
    orig_image = cv2.imread(str(image_path))
    if orig_image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    if use_patch_mode:
        # Patch 模式：划分成多个 patch
        patches, patch_infos = split_image_into_patches(orig_image, patch_size, overlap)
        return patches, patch_infos, orig_image
    else:
        # 原始模式：单张图片处理
        orig_h, orig_w = orig_image.shape[:2]
        
        # 计算缩放比例（保持宽高比，缩放到能放入 input_size x input_size 的方框内）
        scale = min(input_size / orig_h, input_size / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        
        # Resize（保持宽高比）
        resized = cv2.resize(orig_image, (new_w, new_h))
        
        # 创建固定尺寸的画布（padding to input_size x input_size）
        canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        # 居中放置
        pad_top = (input_size - new_h) // 2
        pad_left = (input_size - new_w) // 2
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        
        # 转换为 RGB
        image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # 转换为 PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # 转换为 tensor 并归一化
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(pil_image).unsqueeze(0)  # (1, 3, input_size, input_size)
        
        pad_info = (pad_left, pad_top)
        
        return image_tensor, orig_image, scale, pad_info


@torch.no_grad()
def inference(model, image_tensor, device='cuda', conf_thresh=0.3, feature_extractor=None,
              nms_iou=0.0):
    """
    执行推理
    
    Args:
        model: 模型
        image_tensor: 图片张量 (1, 3, H, W)
        device: 设备
        conf_thresh: 置信度阈值
        feature_extractor: 特征提取器（可选，用于捕获中间层特征）
        nms_iou: NMS IoU阈值（0表示禁用）
    
    Returns:
        predictions: 预测结果列表 [{boxes, labels, scores}, ...]
    """
    image_tensor = image_tensor.to(device)
    
    # 如果需要提取特征，注册 hooks
    if feature_extractor is not None:
        feature_extractor.register_hooks()
    
    # 前向传播
    # 计算模型权重参数的平均值
    weight_params = [p for p in model.parameters() if p.requires_grad and p.data.is_floating_point()]
    if weight_params:
        avg_weight = torch.mean(torch.cat([p.data.flatten() for p in weight_params]))
        print(f"模型权重参数的平均值: {avg_weight.item():.6f}")
    
    # 🔥 判断是否使用 DataParallel
    is_data_parallel = isinstance(model, nn.DataParallel)
    
    if is_data_parallel:
        # 多GPU模式：直接在原始模型上调用（避免 DataParallel 的参数分散问题）
        # 临时取出原始模型进行推理
        original_model = model.module
        original_model.eval()
        outputs = original_model(image_tensor, None, False)  # 🔥 修复：is_teacher=False 避免使用预缓存锚点
    else:
        # 单GPU模式：正常调用
        outputs = model(image_tensor, None, False)  # 🔥 修复：is_teacher=False 避免使用预缓存锚点
    
    # 如果提取了特征，移除 hooks
    if feature_extractor is not None:
        feature_extractor.remove_hooks()
    
    # 后处理
    # 假设模型输出包含 'pred_boxes', 'pred_logits'
    if isinstance(outputs, dict):
        pred_boxes = outputs['pred_boxes']  # (B, num_queries, 4)
        pred_logits = outputs['pred_logits']  # (B, num_queries, num_classes)
    else:
        pred_boxes, pred_logits = outputs
    
    # 获取预测分数和类别
    pred_scores = pred_logits.sigmoid()  # (B, num_queries, num_classes)
    
    predictions = []
    for i in range(pred_boxes.shape[0]):
        # 获取每个 query 的最高分数和对应类别
        scores, labels = pred_scores[i].max(dim=-1)  # (num_queries,)
        boxes_cxcywh = pred_boxes[i]  # (num_queries, 4)
        
        # ========== 双重过滤策略：置信度过滤 + NMS去重 ==========
        # 第1步：置信度过滤
        keep = scores >= conf_thresh
        if keep.sum() == 0:
            top_idx = torch.argmax(scores)
            keep = torch.zeros_like(scores, dtype=torch.bool)
            keep[top_idx] = True
        
        temp_boxes = boxes_cxcywh[keep]
        temp_labels = labels[keep]
        temp_scores = scores[keep]
        
        print(f"   - 置信度过滤：{len(scores)} → {len(temp_scores)} queries")
        
        # 第2步：NMS过滤（去除重叠框）
        if nms_iou > 0 and len(temp_boxes) > 0:
            from torchvision.ops import batched_nms
            
            boxes_xyxy = torch.cat([
                temp_boxes[:, :2] - temp_boxes[:, 2:] / 2,
                temp_boxes[:, :2] + temp_boxes[:, 2:] / 2
            ], dim=-1)
            
            nms_keep = batched_nms(
                boxes=boxes_xyxy,
                scores=temp_scores,
                idxs=temp_labels,
                iou_threshold=nms_iou
            )
            
            filtered_boxes = temp_boxes[nms_keep]
            filtered_labels = temp_labels[nms_keep]
            filtered_scores = temp_scores[nms_keep]
            
            print(f"   - NMS过滤：{len(temp_scores)} → {len(filtered_scores)} queries")
        else:
            filtered_boxes = temp_boxes
            filtered_labels = temp_labels
            filtered_scores = temp_scores
        
        predictions.append({
            'boxes': filtered_boxes.cpu(),
            'labels': filtered_labels.cpu(),
            'scores': filtered_scores.cpu()
        })
    
    return predictions


def merge_patch_predictions(all_predictions, patch_infos, conf_thresh=0.3, nms_iou=0.5):
    """
    合并多个 patch 的预测结果
    
    Args:
        all_predictions: 所有 patch 的预测结果列表，每个元素为 {boxes, labels, scores}
        patch_infos: patch 信息列表，每个元素为 (x_start, y_start, x_end, y_end)
        conf_thresh: 置信度阈值
        nms_iou: NMS IoU 阈值
    
    Returns:
        merged_predictions: 合并后的预测结果 {boxes, labels, scores}
    """
    all_boxes = []
    all_labels = []
    all_scores = []
    
    # 转换每个 patch 的坐标到原图坐标
    for pred, (x_start, y_start, x_end, y_end) in zip(all_predictions, patch_infos):
        boxes = pred['boxes']  # (N, 4) [cx, cy, w, h] normalized to patch
        labels = pred['labels']  # (N,)
        scores = pred['scores']  # (N,)
        
        # 过滤低置信度
        keep = scores >= conf_thresh
        if keep.sum() == 0:
            continue
        
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        # 转换坐标：从 patch 归一化坐标 -> 原图像素坐标
        patch_w = x_end - x_start
        patch_h = y_end - y_start
        
        # 从归一化 [cx, cy, w, h] 转换为 patch 像素坐标 [x1, y1, x2, y2]
        # boxes 是归一化的 [cx, cy, w, h]，范围 [0, 1]
        boxes_xyxy_patch = torch.stack([
            (boxes[:, 0] - boxes[:, 2] / 2) * patch_w,  # x1
            (boxes[:, 1] - boxes[:, 3] / 2) * patch_h,  # y1
            (boxes[:, 0] + boxes[:, 2] / 2) * patch_w,  # x2
            (boxes[:, 1] + boxes[:, 3] / 2) * patch_h   # y2
        ], dim=1)
        
        # 转换到原图坐标
        boxes_xyxy_orig = boxes_xyxy_patch.clone()
        boxes_xyxy_orig[:, [0, 2]] += x_start  # x 坐标
        boxes_xyxy_orig[:, [1, 3]] += y_start  # y 坐标
        
        all_boxes.append(boxes_xyxy_orig)
        all_labels.append(labels)
        all_scores.append(scores)
    
    if len(all_boxes) == 0:
        return {
            'boxes': torch.empty((0, 4)),
            'labels': torch.empty((0,), dtype=torch.long),
            'scores': torch.empty((0,))
        }
    
    # 合并所有结果
    merged_boxes = torch.cat(all_boxes, dim=0)  # (N_total, 4) [x1, y1, x2, y2]
    merged_labels = torch.cat(all_labels, dim=0)  # (N_total,)
    merged_scores = torch.cat(all_scores, dim=0)  # (N_total,)
    
    # 执行 NMS 去除重复检测
    if nms_iou > 0 and len(merged_boxes) > 0:
        from torchvision.ops import batched_nms
        nms_keep = batched_nms(
            boxes=merged_boxes,
            scores=merged_scores,
            idxs=merged_labels,
            iou_threshold=nms_iou
        )
        
        merged_boxes = merged_boxes[nms_keep]
        merged_labels = merged_labels[nms_keep]
        merged_scores = merged_scores[nms_keep]
    
    # 转换回 [cx, cy, w, h] 格式（相对于原图）
    merged_boxes_cxcywh = torch.stack([
        (merged_boxes[:, 0] + merged_boxes[:, 2]) / 2,  # cx
        (merged_boxes[:, 1] + merged_boxes[:, 3]) / 2,  # cy
        merged_boxes[:, 2] - merged_boxes[:, 0],  # w
        merged_boxes[:, 3] - merged_boxes[:, 1]   # h
    ], dim=1)
    
    return {
        'boxes': merged_boxes_cxcywh,
        'labels': merged_labels,
        'scores': merged_scores
    }


def get_random_images_from_directory(directory, num_images=10, extensions=None):
    """
    从指定目录中随机选择指定数量的图片
    
    Args:
        directory: 图片目录路径
        num_images: 要选择的图片数量（默认10）
        extensions: 支持的图片扩展名列表（默认 ['jpg', 'jpeg', 'png', 'bmp']）
    
    Returns:
        selected_images: 选中的图片路径列表
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'bmp']
    
    # 获取所有图片文件
    all_images = []
    for ext in extensions:
        pattern = os.path.join(directory, f'*.{ext}')
        all_images.extend(glob.glob(pattern))
        pattern = os.path.join(directory, f'*.{ext.upper()}')
        all_images.extend(glob.glob(pattern))
    
    if len(all_images) == 0:
        raise ValueError(f"在目录 {directory} 中未找到任何图片文件")
    
    # 去重并排序
    all_images = sorted(list(set(all_images)))
    
    # 随机选择
    num_to_select = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_to_select)
    
    print(f"📁 从 {directory} 中找到 {len(all_images)} 张图片")
    print(f"🎲 随机选择了 {num_to_select} 张图片用于测试")
    
    return selected_images


def draw_predictions(image, predictions, scale=None, pad_info=None, conf_thresh=0.3, input_size=800, is_patch_mode=False):
    """
    在图片上绘制预测结果
    
    Args:
        image: 原始图片（numpy数组，BGR格式）
        predictions: 预测结果 {boxes, labels, scores}
        scale: 缩放比例（非 patch 模式使用）
        pad_info: padding 信息 (pad_left, pad_top)（非 patch 模式使用）
        conf_thresh: 置信度阈值（用于显示）
        input_size: 输入图片尺寸
        is_patch_mode: 是否为 patch 模式（patch 模式下 boxes 已经是原图坐标）
    
    Returns:
        vis_image: 绘制后的图片
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    boxes = predictions['boxes']  # (N, 4) [cx, cy, w, h]
    labels = predictions['labels']  # (N,)
    scores = predictions['scores']  # (N,)
    
    print(f"\n检测到 {len(boxes)} 个目标:")
    print(f"{'ID':<5} {'类别':<20} {'置信度':<10} {'边界框 (x1, y1, x2, y2)'}")
    print(f"{'-'*80}")
    
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < conf_thresh:
            continue
        
        if is_patch_mode:
            # Patch 模式：boxes 已经是原图坐标 [cx, cy, w, h]
            cx, cy, bw, bh = box
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)
        else:
            # 原始模式：需要转换坐标
            if pad_info is None or scale is None:
                raise ValueError("非 patch 模式需要提供 scale 和 pad_info")
            pad_left, pad_top = pad_info
            
            cx, cy, bw, bh = box
            # 从归一化坐标转换到 padding 后的坐标（相对于固定尺寸输入）
            x1_padded = (cx - bw / 2) * input_size
            y1_padded = (cy - bh / 2) * input_size
            x2_padded = (cx + bw / 2) * input_size
            y2_padded = (cy + bh / 2) * input_size
            # 减去 padding，得到相对于缩放后图片的坐标
            x1_scaled = x1_padded - pad_left
            y1_scaled = y1_padded - pad_top
            x2_scaled = x2_padded - pad_left
            y2_scaled = y2_padded - pad_top
            # 缩放回原图
            x1 = int(x1_scaled / scale)
            y1 = int(y1_scaled / scale)
            x2 = int(x2_scaled / scale)
            y2 = int(y2_scaled / scale)
        
        # 边界检查
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # 获取类别名称和颜色
        label_id = int(label.item())
        class_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else f'class_{label_id}'
        color = COLORS[label_id % len(COLORS)]
        
        # 打印信息
        print(f"{idx:<5} {class_name:<20} {score:.4f}    ({x1}, {y1}, {x2}, {y2})")
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label_text = f'{class_name} {score:.2f}'
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        
        # 绘制标签文字
        cv2.putText(vis_image, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    print(f"{'-'*80}\n")
    
    return vis_image


def main():
    parser = argparse.ArgumentParser(description='单张图片测试脚本')
    parser.add_argument('--checkpoint', type=str, 
                       default='/data/DEIMv2/outputs/deimv2_dinov3_x_dji_800/best_stg1.pth',
                       help='检查点文件路径')
    parser.add_argument('--image', type=str, default='/data/DEIMv2/DJI_dataset/images/coco_dataset_1_DJI_20250626102319_0206_patch_1844_3688.jpg',
                       help='测试图片路径')
    parser.add_argument('--config', type=str,
                       default='/data/DEIMv2/configs/deimv2/deimv2_dinov3_x_dji_optimized.yml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default='/data/DEIMv2/visualizations/test_results',
                       help='输出目录')
    parser.add_argument('--conf-thresh', type=float, default=0.30,
                       help='置信度阈值（默认 0.3）')
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='设备（例如 cuda:0、cuda:1 或 cpu）')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='使用的 GPU 数量（>1 时启用 DataParallel）')
    parser.add_argument('--device-ids', type=str, default='5',
                       help='逗号分隔的 GPU ID 列表，例如 "0,1"（仅在 num-gpus>1 时使用）')
    parser.add_argument('--input-size', type=int, default=800,
                       help='输入图片尺寸（默认 800，与训练时一致，非 patch 模式使用）')
    parser.add_argument('--visualize-features', action='store_true',default=True,
                       help='可视化中间层特征（backbone、encoder、decoder）')
    parser.add_argument('--use-patch-mode', action='store_true',default=False,
                       help='启用 patch 模式（用于 8K 大图，将图片划分成多个 patch 进行检测）')
    parser.add_argument('--patch-size', type=int, default=1024,
                       help='Patch 尺寸（默认 None，自动使用 input-size 的值）')
    parser.add_argument('--patch-overlap', type=float, default=0.2,
                       help='Patch 重叠比例（默认 0.2，即 20%% 重叠，避免边界目标被截断）')
    parser.add_argument('--nms-iou', type=float, default=0.5,
                       help='NMS IoU阈值（默认 0.5，0表示禁用）')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='图片目录路径（如果提供，将从该目录读取图片；否则使用--image指定的单张图片）')
    parser.add_argument('--num-random-images', type=int, default=20,
                       help='从目录随机选择的图片数量（默认10，仅在--images-dir模式下有效）')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（用于可重复的随机选择）')
    
    args = parser.parse_args()
    
    # 设置随机种子（如果提供）
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"🎲 设置随机种子: {args.seed}")
    
    # 确定要测试的图片列表
    if args.images_dir is not None:
        # 从目录读取图片模式
        if not os.path.exists(args.images_dir):
            print(f"❌ 错误：图片目录不存在 - {args.images_dir}")
            return
        if not os.path.isdir(args.images_dir):
            print(f"❌ 错误：指定的路径不是目录 - {args.images_dir}")
            return
        
        try:
            test_images = get_random_images_from_directory(
                args.images_dir, 
                num_images=args.num_random_images
            )
            print(f"✅ 已从目录 {args.images_dir} 选择 {len(test_images)} 张图片进行测试\n")
        except Exception as e:
            print(f"❌ 错误：无法从目录选择图片 - {e}")
            return
    else:
        # 单张图片模式
        if args.image is None:
            print(f"❌ 错误：请提供 --image 或 --images-dir 参数")
            return
        if not os.path.exists(args.image):
            print(f"❌ 错误：图片不存在 - {args.image}")
            return
        test_images = [args.image]
    
    # 检查检查点是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ 错误：检查点不存在 - {args.checkpoint}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    requested_device = args.device.lower()
    if requested_device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("⚠️  警告：CUDA 不可用，使用 CPU")
            device = torch.device('cpu')
        else:
            if ':' in requested_device:
                try:
                    primary_gpu = int(requested_device.split(':', 1)[1])
                except ValueError:
                    primary_gpu = 0
            else:
                primary_gpu = 0
            available_gpus = torch.cuda.device_count()
            if primary_gpu >= available_gpus:
                print(f"⚠️  警告：请求的 GPU {primary_gpu} 不存在，仅检测到 {available_gpus} 个 GPU，改用 cuda:0")
                primary_gpu = 0
            device = torch.device(f'cuda:{primary_gpu}')
            torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    multi_gpu = False
    device_ids = None
    if device.type == 'cuda' and args.num_gpus > 1 and torch.cuda.device_count() > 1:
        available_gpus = torch.cuda.device_count()
        if args.device_ids:
            parsed_ids = []
            for item in args.device_ids.split(','):
                item = item.strip()
                if not item:
                    continue
                try:
                    idx = int(item)
                except ValueError:
                    print(f"⚠️  忽略非法 GPU ID: {item}")
                    continue
                if idx >= available_gpus:
                    print(f"⚠️  GPU ID {idx} 超出范围（共有 {available_gpus} 个 GPU），已忽略")
                    continue
                parsed_ids.append(idx)
            if len(parsed_ids) < 2:
                print("⚠️  指定的 GPU ID 少于 2 个，保持单卡运行")
                device_ids = [device.index or 0]
            else:
                device_ids = parsed_ids[:args.num_gpus]
        else:
            requested = min(args.num_gpus, available_gpus)
            device_ids = list(range(requested))
        
        if device_ids and len(device_ids) > 1:
            primary_gpu = device_ids[0]
            if device.index != primary_gpu:
                device = torch.device(f'cuda:{primary_gpu}')
                torch.cuda.set_device(device)
            multi_gpu = True
            print(f"⚡ 启用多 GPU 推理，使用设备: {device_ids}")
    
    print(f"\n{'='*80}")
    if args.images_dir is not None:
        print("🎯 批量测试（从目录读取）")
    else:
        print("🎯 单张图片测试")
    print(f"{'='*80}")
    if args.images_dir is not None:
        print(f"图片目录: {args.images_dir}")
        print(f"测试图片数: {len(test_images)}")
    else:
        print(f"图片: {args.image}")
    print(f"检查点: {args.checkpoint}")
    print(f"置信度阈值: {args.conf_thresh}")
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    # 加载模型
    model, cfg = load_model(args.checkpoint, args.config, device)
    
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        model.eval()
        print(f"✅ 模型已包装为 DataParallel（设备: {device_ids}）")
    
    # 批量处理所有图片
    use_patch_mode = args.use_patch_mode
    
    # 如果使用 patch 模式，patch_size 未指定时使用 input_size
    if use_patch_mode:
        patch_size = args.patch_size if args.patch_size is not None else args.input_size
    else:
        patch_size = args.input_size  # 非 patch 模式不使用，但为了代码一致性
    
    # 处理每张图片
    for img_idx, image_path in enumerate(test_images, 1):
        print(f"\n{'='*80}")
        print(f"📸 处理图片 {img_idx}/{len(test_images)}: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        
        # 预处理图片
        print("📸 预处理图片...")
        
        if use_patch_mode:
            # Patch 模式：划分成多个 patch
            patches, patch_infos, orig_image = preprocess_image(
                image_path, 
                args.input_size, 
                use_patch_mode=True,
                patch_size=patch_size,
                overlap=args.patch_overlap
            )
            print(f"   - 原始尺寸: {orig_image.shape[:2]}")
            print(f"   - Patch 尺寸: {patch_size}×{patch_size}")
            print(f"   - 重叠比例: {args.patch_overlap*100:.1f}%")
            print(f"   - 总 Patch 数: {len(patches)}\n")
            
            # 执行推理（对每个 patch）
            print("🚀 执行推理（Patch 模式）...")
            all_predictions = []
            
            for i, patch in enumerate(patches):
                print(f"   处理 Patch {i+1}/{len(patches)}...", end=' ')
                patch_tensor = preprocess_patch(patch, patch_size)
                
                # 对单个 patch 进行推理（不使用特征提取，因为 patch 模式不支持）
                patch_pred = inference(
                    model=model,
                    image_tensor=patch_tensor,
                    device=device,
                    conf_thresh=args.conf_thresh,
                    feature_extractor=None,  # Patch 模式不支持特征提取
                    nms_iou=0.0  # 先不进行 NMS，最后统一合并时再 NMS
                )
                all_predictions.append(patch_pred[0])
                print(f"检测到 {len(patch_pred[0]['boxes'])} 个目标")
            
            # 合并所有 patch 的预测结果
            print(f"\n🔄 合并 {len(all_predictions)} 个 patch 的预测结果...")
            merged_predictions = merge_patch_predictions(
                all_predictions, 
                patch_infos, 
                conf_thresh=args.conf_thresh,
                nms_iou=args.nms_iou
            )
            print(f"   合并后共 {len(merged_predictions['boxes'])} 个目标\n")
            
            predictions = [merged_predictions]
        else:
            # 原始模式：单张图片处理
            image_tensor, orig_image, scale, pad_info = preprocess_image(
                image_path, 
                args.input_size, 
                use_patch_mode=False
            )
            pad_left, pad_top = pad_info
            print(f"   - 原始尺寸: {orig_image.shape[:2]}")
            print(f"   - 输入尺寸: {image_tensor.shape[2:]}")
            print(f"   - 缩放比例: {scale:.4f}")
            print(f"   - Padding: left={pad_left}, top={pad_top}\n")
            
            # 执行推理
            print("🚀 执行推理...")
            
            enable_feature_vis = args.visualize_features and (img_idx == 1)  # 只对第一张图片可视化特征
            if multi_gpu and enable_feature_vis:
                print("⚠️  当前为多 GPU 模式，暂不支持特征可视化，自动禁用")
                enable_feature_vis = False
            
            # 如果需要可视化特征，创建特征提取器
            feature_extractor = None
            if enable_feature_vis:
                hook_model = model.module if isinstance(model, nn.DataParallel) else model
                feature_extractor = FeatureExtractor(hook_model)
                print("   ✅ 已启用特征提取")
            
            predictions = inference(
                model=model, 
                image_tensor=image_tensor, 
                device=device, 
                conf_thresh=args.conf_thresh, 
                feature_extractor=feature_extractor,
                nms_iou=args.nms_iou
            )
        
        # 绘制结果
        print("🎨 绘制预测结果...")
        if use_patch_mode:
            vis_image = draw_predictions(
                orig_image, 
                predictions[0], 
                conf_thresh=args.conf_thresh, 
                input_size=patch_size,  # Patch 模式使用 patch_size
                is_patch_mode=True
            )
        else:
            vis_image = draw_predictions(
                orig_image, 
                predictions[0], 
                scale, 
                pad_info, 
                args.conf_thresh, 
                args.input_size,
                is_patch_mode=False
            )
        
        # 保存结果
        image_name = Path(image_path).stem
        output_path = output_dir / f"{image_name}_result.jpg"
        cv2.imwrite(str(output_path), vis_image)
        
        print(f"✅ 结果已保存到: {output_path}")
        
        # 如果启用了特征可视化且是第一张图片，保存特征图
        if enable_feature_vis and feature_extractor is not None:
            features = feature_extractor.get_features()
            if features:
                visualize_feature_maps(features, orig_image, output_dir, image_name)
    
    print(f"\n{'='*80}")
    if args.images_dir is not None:
        print(f"🎉 批量测试完成！共处理 {len(test_images)} 张图片")
    else:
        print("🎉 测试完成！")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

