"""
语义原型提取脚本
从已训练的全监督模型中提取每个类别的语义特征（dec_hs）

用法:
    python extract_semantic_prototypes.py \
        --config configs/deim/coco/ssod_deimv2_r50_coco.yml \
        --checkpoint outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth \
        --output semantic_prototypes.pth \
        --device cuda:0 \
        --score-thresh 0.7 \
        --iou-thresh 0.5 \
        --max-batches 100
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from engine.core import YAMLConfig
from engine.misc import dist_utils


def box_cxcywh_to_xyxy(boxes, image_size):
    """
    将 cxcywh (归一化) 格式转换为 xyxy (绝对坐标)
    
    Args:
        boxes: [N, 4] in [0,1], format: (cx, cy, w, h)
        image_size: (H, W) or (W, H) - 会自动处理
    
    Returns:
        boxes_xyxy: [N, 4] in pixel coordinates
    """
    if len(image_size) == 2:
        H, W = image_size
    else:
        W, H = image_size[0], image_size[1]
    
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w/2) * W
    y1 = (cy - h/2) * H
    x2 = (cx + w/2) * W
    y2 = (cy + h/2) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_xyxy(boxes1, boxes2):
    """
    计算两组边界框的 IoU
    
    Args:
        boxes1: [N, 4] xyxy format
        boxes2: [M, 4] xyxy format
    
    Returns:
        iou: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def load_model_and_checkpoint(config_path, checkpoint_path, device):
    """
    加载模型和检查点
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 检查点文件路径
        device: 设备
    
    Returns:
        model: 加载好的模型
        cfg: 配置对象
    """
    print(f"\n{'='*80}")
    print("📦 加载模型...")
    print(f"{'='*80}")
    print(f"配置文件: {config_path}")
    print(f"检查点: {checkpoint_path}")
    print(f"设备: {device}\n")
    
    # 加载配置
    cfg = YAMLConfig(config_path, resume=None)
    
    # 创建模型
    model = cfg.model
    model = model.to(device)
    model.eval()
    
    # 加载权重
    print("加载检查点权重...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理不同的检查点格式
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best AP: {checkpoint.get('best_ap', 'unknown')}")
    else:
        state_dict = checkpoint
    
    # 移除 'module.' 前缀（如果有）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if missing:
        print(f"⚠️  缺失的键 ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"⚠️  意外的键 ({len(unexpected)}): {unexpected[:3]}...")
    
    print("✅ 模型加载成功！\n")
    
    return model, cfg


@torch.no_grad()
def extract_class_prototypes(
    model,
    dataloader,
    device,
    num_classes=80,
    score_thresh=0.0,  # 改为0.0，不过滤（因为我们直接用GT）
    iou_thresh=0.5,
    max_batches=None,
    verbose=True,
    use_gt_directly=True  # 新参数：是否直接使用GT（推荐）
):
    """
    从标注数据中提取每个类别的语义原型
    
    Args:
        model: 训练好的模型
        dataloader: 标注数据的 dataloader
        device: 设备
        num_classes: 类别数量
        score_thresh: 置信度阈值（仅在 use_gt_directly=False 时使用）
        iou_thresh: IoU 阈值（用于匹配 GT 和 query）
        max_batches: 最大处理批次数（None 表示全部）
        verbose: 是否显示详细信息
        use_gt_directly: 是否直接使用GT框找对应query（推荐True）
    
    Returns:
        prototypes_dict: 包含每个类别特征的字典
    """
    model.eval()
    
    # 存储每个类别的特征
    class_features = defaultdict(list)  # {class_id: [features]}
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    
    # 统计每个类别的GT总数和匹配数
    total_gt_per_class = defaultdict(int)
    matched_gt_per_class = defaultdict(int)
    filtered_gt_per_class = defaultdict(int)  # 被过滤掉的GT数量
    
    print(f"\n{'='*80}")
    print("🔍 提取语义特征...")
    print(f"{'='*80}")
    print(f"类别数量: {num_classes}")
    print(f"提取模式: {'基于GT（推荐）' if use_gt_directly else f'基于预测（置信度>{score_thresh}）'}")
    print(f"IoU 阈值: {iou_thresh}")
    print(f"最大批次: {max_batches if max_batches else '全部'}\n")
    
    total_samples = 0
    total_gt_objects = 0  # 总GT数量
    total_matched = 0
    
    # 遍历数据集
    pbar = tqdm(enumerate(dataloader), total=len(dataloader) if max_batches is None else max_batches,
                desc="提取特征", disable=not verbose)
    
    for batch_idx, batch in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        # 准备输入
        images = batch[0].to(device)  # [B, 3, H, W]
        targets = batch[1]  # list of dicts
        
        B = images.size(0)
        total_samples += B
        
        # 模型前向传播
        outputs = model(images, is_teacher=True)
        
        # 提取输出
        pred_logits = outputs['pred_logits']  # [B, Q_matching, C]
        pred_boxes = outputs['pred_boxes']    # [B, Q_matching, 4] (cxcywh norm)
        dec_hs = outputs.get('dec_hs')        # {0~5: [B, Q_total, 256]}
        
        if dec_hs is None:
            print("⚠️  模型输出中没有 dec_hs，跳过...")
            continue
        
        # 获取最后一层的特征
        last_layer_idx = max(dec_hs.keys()) if isinstance(dec_hs, dict) else len(dec_hs) - 1
        features_all = dec_hs[last_layer_idx]  # [B, Q_total, 256]
        
        # 对齐 DN 查询：dec_hs 包含所有查询，pred_logits 只包含匹配查询
        total_queries = features_all.shape[1]
        matching_queries = pred_logits.shape[1]
        dn_queries = total_queries - matching_queries
        
        # 提取匹配查询的特征（后 matching_queries 个）
        features = features_all[:, dn_queries:, :]  # [B, Q_matching, 256]
        
        # 确保形状对齐
        assert features.shape[1] == pred_logits.shape[1], \
            f"特征形状不匹配: {features.shape[1]} vs {pred_logits.shape[1]}"
        
        # 计算预测分数和类别（用于参考，不一定用于过滤）
        probs = pred_logits.sigmoid()  # [B, Q, C]
        scores, pred_classes = probs.max(dim=-1)  # [B, Q]
        
        # 逐样本处理
        batch_matched = 0
        for b in range(B):
            # 获取 GT
            gt_labels = targets[b]['labels'].to(device)  # [N_gt]
            gt_boxes = targets[b]['boxes'].to(device)  # [N_gt, 4] cxcywh norm
            
            if len(gt_boxes) == 0:
                continue
            
            total_gt_objects += len(gt_boxes)
            
            # 统计每个类别的GT总数
            for label in gt_labels:
                total_gt_per_class[label.item()] += 1
            
            # 转换为绝对坐标
            H, W = images.shape[2], images.shape[3]
            
            if use_gt_directly:
                # ============ 推荐模式：直接用GT框找对应query ============
                # 对每个GT框，找到IoU最高的query（不管置信度）
                
                all_pred_boxes = pred_boxes[b]  # [Q, 4] cxcywh norm
                all_features = features[b]  # [Q, 256]
                
                pred_xyxy = box_cxcywh_to_xyxy(all_pred_boxes, (H, W))
                gt_xyxy = box_cxcywh_to_xyxy(gt_boxes, (H, W))
                
                # 计算 IoU: [Q, N_gt]
                ious = box_iou_xyxy(pred_xyxy, gt_xyxy)
                
                # 对每个GT，找最匹配的query
                max_iou_per_gt, best_query_idx = ious.max(dim=0)  # [N_gt]
                
                # 只保留IoU >= 阈值的GT
                valid_gt = max_iou_per_gt >= iou_thresh
                
                # 统计被过滤掉的GT（按类别）
                filtered_gt = ~valid_gt
                if filtered_gt.any():
                    filtered_labels = gt_labels[filtered_gt]
                    for label in filtered_labels:
                        filtered_gt_per_class[label.item()] += 1
                
                if valid_gt.sum() == 0:
                    continue
                
                # 提取对应的特征
                matched_features = all_features[best_query_idx[valid_gt]]  # [N_valid, 256]
                matched_labels = gt_labels[valid_gt]  # [N_valid]
                matched_ious = max_iou_per_gt[valid_gt]  # [N_valid]（用于调试）
                
                # L2 归一化
                matched_features = F.normalize(matched_features, dim=-1)
                
                # 按类别存储
                for feat, label, iou in zip(matched_features, matched_labels, matched_ious):
                    class_id = label.item()
                    class_features[class_id].append(feat.cpu())
                    class_counts[class_id] += 1
                    matched_gt_per_class[class_id] += 1
                
                batch_matched += valid_gt.sum().item()
        
        total_matched += batch_matched
        
        # 更新进度条
        if verbose:
            pbar.set_postfix({
                'matched': f'{total_matched}/{total_gt_objects}',
                'coverage': f'{total_matched/max(total_gt_objects, 1)*100:.1f}%',
                'classes': len(class_features)
            })
    
    print(f"\n✅ 特征提取完成！")
    print(f"  总样本数（图片）: {total_samples}")
    print(f"  总GT物体数: {total_gt_objects}")
    print(f"  成功匹配数: {total_matched}")
    print(f"  匹配覆盖率: {total_matched/max(total_gt_objects, 1)*100:.1f}%")
    print(f"  有特征的类别数: {len(class_features)}/{num_classes}\n")
    
    # 统计被过滤的GT数量
    total_filtered = sum(filtered_gt_per_class.values())
    if total_filtered > 0:
        print(f"⚠️  被过滤的GT统计：")
        print(f"  总过滤数: {total_filtered} ({total_filtered/max(total_gt_objects, 1)*100:.1f}%)")
        print(f"  原因: IoU < {iou_thresh}（模型未准确检测到这些GT）\n")
    
    # 统计每个类别的特征数量和覆盖率
    print("各类别统计（GT总数 vs 成功匹配 vs 覆盖率）：")
    print("-" * 80)
    print(f"{'类别ID':>7} | {'GT总数':>7} | {'匹配数':>7} | {'过滤数':>7} | {'覆盖率':>7}")
    print("-" * 80)
    
    for class_id in sorted(total_gt_per_class.keys()):
        total_gt = total_gt_per_class[class_id]
        matched = matched_gt_per_class[class_id]
        filtered = filtered_gt_per_class.get(class_id, 0)
        coverage = matched / total_gt * 100 if total_gt > 0 else 0.0
        
        # 如果覆盖率 < 80%，用 ⚠️ 标记
        warning = "⚠️ " if coverage < 80 else "   "
        
        print(f"{warning}{class_id:>4} | {total_gt:>7} | {matched:>7} | {filtered:>7} | {coverage:>6.1f}%")
    
    print("-" * 80)
    
    # 汇总低覆盖率类别
    low_coverage_classes = []
    for class_id in sorted(total_gt_per_class.keys()):
        total_gt = total_gt_per_class[class_id]
        matched = matched_gt_per_class[class_id]
        coverage = matched / total_gt * 100 if total_gt > 0 else 0.0
        if coverage < 80 and total_gt >= 10:  # 只统计样本数>=10的类别
            low_coverage_classes.append((class_id, coverage, total_gt, matched))
    
    if low_coverage_classes:
        print(f"\n⚠️  低覆盖率类别（< 80%）：")
        for class_id, coverage, total_gt, matched in low_coverage_classes:
            print(f"  类别 {class_id:2d}: {coverage:.1f}% ({matched}/{total_gt})")
        print(f"\n说明：这些类别的部分GT因为IoU<{iou_thresh}被过滤（可能是小目标、遮挡或模型漏检）")
    
    print("-" * 80)
    
    # 转换为张量并计算均值原型
    prototypes = torch.zeros(num_classes, 256)
    prototype_stds = torch.zeros(num_classes, 256)
    
    for class_id, feats in class_features.items():
        if len(feats) > 0:
            feats_tensor = torch.stack(feats)  # [N, 256]
            prototypes[class_id] = feats_tensor.mean(dim=0)
            prototypes[class_id] = F.normalize(prototypes[class_id], dim=-1)
            
            # 计算标准差（可选）
            prototype_stds[class_id] = feats_tensor.std(dim=0)
    
    # 计算类内相似度统计
    print("\n类内相似度统计：")
    print("-" * 60)
    class_similarities = {}
    for class_id, feats in class_features.items():
        if len(feats) > 10:  # 至少10个样本才统计
            feats_tensor = torch.stack(feats)
            prototype = prototypes[class_id].unsqueeze(0)
            
            # 计算每个特征与原型的余弦相似度
            sims = F.cosine_similarity(feats_tensor, prototype, dim=-1)
            
            class_similarities[class_id] = {
                'mean': sims.mean().item(),
                'std': sims.std().item(),
                'min': sims.min().item(),
                'max': sims.max().item(),
                'q25': sims.quantile(0.25).item(),
                'q50': sims.quantile(0.50).item(),
                'q75': sims.quantile(0.75).item(),
            }
            
            print(f"  类别 {class_id:2d}: "
                  f"mean={sims.mean():.3f}, "
                  f"std={sims.std():.3f}, "
                  f"min={sims.min():.3f}, "
                  f"q25={sims.quantile(0.25):.3f}, "
                  f"q50={sims.quantile(0.50):.3f}, "
                  f"q75={sims.quantile(0.75):.3f}")
    print("-" * 60)
    
    return {
        'prototypes': prototypes,  # [num_classes, 256] 每类的均值原型
        'counts': class_counts,  # [num_classes] 每类的特征数量
        'stds': prototype_stds,  # [num_classes, 256] 每类的标准差
        'similarities': class_similarities,  # {class_id: stats_dict}
        'all_features': {k: torch.stack(v) for k, v in class_features.items()},  # {class_id: [N, 256]} 保存所有的原始特征
    }


def main():
    parser = argparse.ArgumentParser(description='提取语义原型')
    parser.add_argument('--config', type=str, default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default='outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth',
                        help='检查点文件路径')
    parser.add_argument('--output', type=str, default='semantic_prototypes.pth',
                        help='输出文件路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备 (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--score-thresh', type=float, default=0.0,
                        help='置信度阈值（仅在 --no-use-gt 时使用，建议0.6-0.8）')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='IoU 阈值（GT与query匹配的最小IoU）')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='最大处理批次数（用于快速测试）')
    parser.add_argument('--num-classes', type=int, default=80,
                        help='类别数量')
    parser.add_argument('--no-use-gt', action='store_true',
                        help='不直接使用GT（使用旧的基于预测的模式）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, cfg = load_model_and_checkpoint(args.config, args.checkpoint, device)
    
    # ⚠️ 关键修复：设置动态位置编码和动态anchor生成，避免固定尺寸导致的维度不匹配
    # 参考 test_single_image.py 的做法
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'eval_spatial_size'):
        original_encoder_size = model.encoder.eval_spatial_size
        model.encoder.eval_spatial_size = None
        print(f"✅ 已设置 Encoder 动态位置编码（eval_spatial_size: {original_encoder_size} → None）")
    
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'eval_spatial_size'):
        original_decoder_size = model.decoder.eval_spatial_size
        model.decoder.eval_spatial_size = None
        print(f"✅ 已设置 Decoder 动态anchor生成（eval_spatial_size: {original_decoder_size} → None）")
    
    print("   这样可以处理不同尺寸的图像，避免尺寸不匹配错误\n")
    
    # 获取标注数据的 dataloader
    print("加载数据集...")
    
    # 使用训练数据的 dataloader（标注数据）
    dataloader = cfg.train_dataloader
    
    if dataloader is None:
        print("❌ 无法从配置中加载数据集！")
        print("   请检查配置文件中的 train_dataloader 设置")
        return
    
    print(f"✅ 数据集加载成功！")
    print(f"   批次大小: {dataloader.batch_size}")
    print(f"   总批次数: {len(dataloader)}")
    print(f"   总样本数: ~{len(dataloader) * dataloader.batch_size}\n")
    
    # 提取原型
    prototypes_dict = extract_class_prototypes(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=args.num_classes,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
        max_batches=args.max_batches,
        verbose=True,
        use_gt_directly=not args.no_use_gt  # 默认使用GT模式（推荐）
    )
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存基本信息（不包含 all_features，太大）
    save_dict = {
        'prototypes': prototypes_dict['prototypes'],
        'counts': prototypes_dict['counts'],
        'stds': prototypes_dict['stds'],
        'similarities': prototypes_dict['similarities'],
        'all_features': prototypes_dict['all_features'],
        'config': {
            'checkpoint': args.checkpoint,
            'score_thresh': args.score_thresh,
            'iou_thresh': args.iou_thresh,
            'num_classes': args.num_classes,
        }
    }
    
    torch.save(save_dict, output_path)
    print(f"\n✅ 语义原型已保存到: {output_path}")
    all_features_path = output_path.with_stem(output_path.stem + '_all_features')
    torch.save(prototypes_dict['all_features'], all_features_path)
    print(f"✅ 所有特征已保存到: {all_features_path}")
    
    # 打印使用建议
    print(f"\n{'='*80}")
    print("📊 使用建议：")
    print(f"{'='*80}")
    print("1. 查看类内相似度统计，如果某个类别的 std 很大（>0.2），")
    print("   说明类内差异大，建议使用多原型方案")
    print()
    print("2. 加载原型使用示例：")
    print(f"   >>> prototypes = torch.load('{output_path}')")
    print(f"   >>> semantic_features = prototypes['prototypes']  # [80, 256]")
    print(f"   >>> class_counts = prototypes['counts']  # [80]")
    print()
    print("3. 如果需要多原型，可以对 all_features 做 K-Means 聚类：")
    print(f"   python extract_semantic_prototypes.py ... --save-all-features")
    print()
    print("4. 提取模式说明：")
    print("   - 默认模式（推荐）：直接用GT框找对应query，覆盖所有样本")
    print("   - 旧模式：python extract_semantic_prototypes.py ... --no-use-gt --score-thresh 0.7")
    print("             （只用高置信度预测，可能遗漏困难样本）")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

