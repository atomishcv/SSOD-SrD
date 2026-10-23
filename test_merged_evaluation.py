"""
测试脚本：使用合并类别评估方式测试模型精度

使用方法:
    python test_merged_evaluation.py \
        --config configs/deimv2/deimv2_dinov3_x_dji_optimized.yml \
        --checkpoint outputs/deimv2_dinov3_x_dji_800/best_stg1.pth \
        --ann-file DJI_dataset/annotations/val_full_two_categories.json \
        --nms-iou 0.5
"""

import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))

from engine.core import YAMLConfig
from engine.solver.det_engine import evaluate_merged_classes
from engine.misc import dist_utils
from faster_coco_eval import COCO


def load_model_from_checkpoint(checkpoint_path: str, cfg, device: torch.device):
    """
    从检查点加载模型
    
    Args:
        checkpoint_path: 检查点文件路径
        cfg: 配置对象
        device: 设备
    
    Returns:
        model: 加载好的模型
    """
    print(f"\n{'='*80}")
    print("📦 加载模型...")
    print(f"{'='*80}")
    print(f"检查点: {checkpoint_path}")
    print(f"设备: {device}\n")
    
    # 创建模型
    model = cfg.model.to(device)
    model.eval()
    
    # 加载权重
    print("加载检查点权重...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 处理不同的检查点格式
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("✅ 使用 'model' 键中的权重")
    elif 'ema' in checkpoint:
        # 如果有 EMA 权重，优先使用
        ema_state = checkpoint['ema']
        if isinstance(ema_state, dict) and 'module' in ema_state:
            state_dict = ema_state['module']
            print("✅ 使用 EMA 模型权重 (ema.module)")
        else:
            state_dict = ema_state
            print("✅ 使用 EMA 模型权重 (ema)")
    else:
        state_dict = checkpoint
        print("✅ 直接加载 state_dict")
    
    # 移除 'module.' 前缀（如果有，处理 DDP 包装）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # 加载权重（使用 strict=False 以兼容部分不匹配的情况）
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"⚠️  缺失的键（已忽略）: {len(missing_keys)} 个")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"   - {key}")
        else:
            for key in missing_keys[:10]:
                print(f"   - {key}")
            print(f"   ... 还有 {len(missing_keys) - 10} 个")
    
    if unexpected_keys:
        print(f"⚠️  意外的键（已忽略）: {len(unexpected_keys)} 个")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"   - {key}")
        else:
            for key in unexpected_keys[:10]:
                print(f"   - {key}")
            print(f"   ... 还有 {len(unexpected_keys) - 10} 个")
    
    # 设置动态位置编码（如果模型支持）
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'eval_spatial_size'):
        model.encoder.eval_spatial_size = None
        print("✅ 已设置动态位置编码（eval_spatial_size=None）")
    
    print(f"✅ 模型加载完成\n")
    return model


def load_coco_gt_from_json(ann_file: str):
    """
    从 JSON 文件加载 COCO GT 对象
    
    Args:
        ann_file: 标注文件路径
    
    Returns:
        coco_gt: COCO GT 对象
    """
    print(f"📂 加载标注文件: {ann_file}")
    
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"标注文件不存在: {ann_file}")
    
    coco_gt = COCO(ann_file)
    
    # 打印数据集信息
    print(f"✅ 标注文件加载完成")
    print(f"   - 图像数量: {len(coco_gt.imgs)}")
    print(f"   - 标注数量: {len(coco_gt.anns)}")
    print(f"   - 类别数量: {len(coco_gt.cats)}")
    print(f"   - 类别列表: {[cat['name'] for cat in coco_gt.cats.values()]}\n")
    
    return coco_gt


def main():
    parser = argparse.ArgumentParser(description='使用合并类别评估方式测试模型精度')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/deimv2_dinov3_x_dji_800/best_stg1.pth',
        help='模型检查点路径'
    )
    parser.add_argument(
        '--ann-file',
        type=str,
        default='DJI_dataset/annotations/val_full_two_categories.json',
        help='验证集标注文件路径'
    )
    parser.add_argument(
        '--nms-iou',
        type=float,
        default=0.5,
        help='合并评估时 NMS 的 IoU 阈值（默认: 0.5）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='设备 (cuda 或 cpu)'
    )
    parser.add_argument(
        '--use-hsv',
        action='store_true',
        default=True,
        help='是否使用 HSV 输入（如果模型支持）'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    # 设置设备
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，使用 CPU")
        device = torch.device('cpu')
    
    # 初始化分布式（单GPU模式）
    dist_utils.setup_distributed(print_rank=0, print_method='builtin', seed=0)
    
    # 加载配置
    print(f"\n{'='*80}")
    print("📋 加载配置...")
    print(f"{'='*80}")
    print(f"配置文件: {args.config}\n")
    
    cfg = YAMLConfig(args.config, resume=None)
    
    # 修改验证数据加载器的标注文件路径
    if hasattr(cfg, 'val_dataloader') and hasattr(cfg.val_dataloader, 'dataset'):
        cfg.val_dataloader.dataset.ann_file = args.ann_file
        print(f"✅ 已设置验证集标注文件: {args.ann_file}")
    
    # 加载模型
    model = load_model_from_checkpoint(args.checkpoint, cfg, device)
    
    # 加载 COCO GT
    coco_gt = load_coco_gt_from_json(args.ann_file)
    
    # 创建验证数据加载器
    print(f"\n{'='*80}")
    print("📊 创建验证数据加载器...")
    print(f"{'='*80}\n")
    
    val_dataloader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    print(f"✅ 验证数据加载器创建完成")
    print(f"   - Batch size: {cfg.val_dataloader.total_batch_size if hasattr(cfg.val_dataloader, 'total_batch_size') else 'N/A'}")
    print(f"   - 数据集大小: {len(val_dataloader.dataset)}\n")
    
    # 创建后处理器和损失函数（用于保持接口一致）
    postprocessor = cfg.postprocessor
    criterion = cfg.criterion.to(device)
    criterion.eval()
    
    # 执行合并类别评估
    print(f"\n{'='*80}")
    print("🚀 开始评估（合并类别方式）...")
    print(f"{'='*80}")
    print(f"评估方式: 合并类别评估（small+big → object + NMS）")
    print(f"NMS IoU 阈值: {args.nms_iou}")
    print(f"使用 HSV: {args.use_hsv}\n")
    
    test_stats, coco_evaluator = evaluate_merged_classes(
        model=model,
        criterion=criterion,
        postprocessor=postprocessor,
        data_loader=val_dataloader,
        coco_gt=coco_gt,
        device=device,
        use_hsv=args.use_hsv,
        nms_iou_threshold=args.nms_iou
    )
    
    # 打印评估结果
    print(f"\n{'='*80}")
    print("📊 评估结果")
    print(f"{'='*80}")
    
    if 'coco_eval_bbox' in test_stats:
        stats = test_stats['coco_eval_bbox']
        if len(stats) >= 12:
            print(f"\nCOCO 评估指标 (合并类别):")
            print(f"  AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]:.4f}")
            print(f"  AP @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]:.4f}")
            print(f"  AP @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]:.4f}")
            print(f"  AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]:.4f}")
            print(f"  AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]:.4f}")
            print(f"  AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]:.4f}")
            print(f"  AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats[6]:.4f}")
            print(f"  AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats[7]:.4f}")
            print(f"  AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]:.4f}")
            print(f"  AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[9]:.4f}")
            print(f"  AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[10]:.4f}")
            print(f"  AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[11]:.4f}")
            
            print(f"\n🎯 主要指标:")
            print(f"  AP@[0.5:0.95] (mAP): {stats[0]:.4f}")
            print(f"  AP@0.5:         {stats[1]:.4f}")
            print(f"  AP@0.75:        {stats[2]:.4f}")
        else:
            print(f"⚠️  评估结果格式异常，统计数据数量: {len(stats)}")
            print(f"   结果: {stats}")
    else:
        print("⚠️  未找到评估结果")
    
    print(f"\n{'='*80}")
    print("✅ 评估完成")
    print(f"{'='*80}\n")
    
    dist_utils.cleanup()


if __name__ == '__main__':
    main()

