"""
监督预训练脚本 - 训练 Encoder 学习类别代表性特征

功能：
1. 在有标注数据上训练完整的目标检测模型
2. 重点优化 encoder 的类别特征表示能力
3. 提取并保存每个类别的代表性特征（原型）
4. 使用 early stopping（验证精度提升幅度小时自动停止）
5. 后续用于评估教师模型伪标签质量（通过余弦相似度）

使用方式：
    python supervised_pretrain.py -c configs/deimv2_ssod/deimv2_dinov3_x_coco.yml
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch.multiprocessing as mp

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS
from engine.solver.det_engine import train_one_epoch, evaluate
from engine.optim import ModelEMA

# NCCL 超时设置
os.environ.setdefault('NCCL_TIMEOUT', '10800')
os.environ.setdefault('NCCL_DEBUG', 'WARN')


class FeatureExtractor:
    """
    特征提取器：从 encoder 输出中提取类别特征
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.class_features = defaultdict(list)  # {class_id: [features]}
        
    @torch.no_grad()
    def extract_features_from_batch(self, images, targets):
        """
        从一个 batch 中提取特征
        
        Args:
            images: 输入图像 (B, 3, H, W)
            targets: 标注信息 [{boxes, labels}, ...]
        
        Returns:
            features_dict: {class_id: features}
        """
        self.model.eval()
        images = images.to(self.device)
        
        # 前向传播到 encoder
        # 假设模型结构：backbone -> encoder -> decoder
        with torch.no_grad():
            # 获取 backbone 特征
            if hasattr(self.model, 'module'):
                model = self.model.module
            else:
                model = self.model
            
            # Backbone
            backbone_features = model.backbone(images)
            
            # Encoder（关键特征提取）
            if hasattr(model, 'encoder'):
                encoder_features = model.encoder(backbone_features)
            else:
                encoder_features = backbone_features
            
            # 如果 encoder 返回多尺度特征，取最后一层
            if isinstance(encoder_features, (list, tuple)):
                encoder_features = encoder_features[-1]  # (B, C, H, W)
            
            # 提取每个目标的特征
            batch_size = images.shape[0]
            features_dict = defaultdict(list)
            
            for idx in range(batch_size):
                target = targets[idx]
                if 'boxes' not in target or 'labels' not in target:
                    continue
                
                boxes = target['boxes']  # (N, 4) [cx, cy, w, h] normalized
                labels = target['labels']  # (N,)
                
                if len(boxes) == 0:
                    continue
                
                feat = encoder_features[idx]  # (C, H, W)
                C, H, W = feat.shape
                
                # 对每个 box 提取 RoI 特征
                for box, label in zip(boxes, labels):
                    label_id = int(label.item())
                    if label_id < 0:
                        continue
                    
                    # 将归一化坐标转换为特征图坐标
                    cx, cy, bw, bh = box.tolist()
                    x1 = max(0, int((cx - bw/2) * W))
                    y1 = max(0, int((cy - bh/2) * H))
                    x2 = min(W, int((cx + bw/2) * W))
                    y2 = min(H, int((cy + bh/2) * H))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # 提取 RoI 特征并池化
                    roi_feat = feat[:, y1:y2, x1:x2]  # (C, h, w)
                    
                    # 自适应平均池化到固定大小
                    roi_feat = F.adaptive_avg_pool2d(roi_feat.unsqueeze(0), (1, 1))  # (1, C, 1, 1)
                    roi_feat = roi_feat.squeeze().cpu()  # (C,)
                    
                    # L2 归一化（用于余弦相似度计算）
                    roi_feat = F.normalize(roi_feat, p=2, dim=0)
                    
                    features_dict[label_id].append(roi_feat)
            
            return features_dict
    
    def accumulate_features(self, features_dict):
        """累积特征"""
        for class_id, features in features_dict.items():
            self.class_features[class_id].extend(features)
    
    def compute_class_prototypes(self):
        """
        计算每个类别的原型特征（特征中心）
        
        Returns:
            prototypes: {class_id: prototype_feature}
            stats: {class_id: {'count': int, 'std': float}}
        """
        prototypes = {}
        stats = {}
        
        for class_id, features in self.class_features.items():
            if len(features) == 0:
                continue
            
            # 堆叠特征
            features_tensor = torch.stack(features)  # (N, C)
            
            # 计算原型（均值）
            prototype = features_tensor.mean(dim=0)  # (C,)
            prototype = F.normalize(prototype, p=2, dim=0)  # L2 归一化
            
            # 计算标准差（特征分散程度）
            std = features_tensor.std(dim=0).mean().item()
            
            prototypes[class_id] = prototype
            stats[class_id] = {
                'count': len(features),
                'std': std
            }
        
        return prototypes, stats
    
    def clear(self):
        """清空累积的特征"""
        self.class_features.clear()


class EarlyStopping:
    """
    Early Stopping 机制
    
    当验证指标（AP）提升幅度小于阈值时停止训练
    """
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        """
        Args:
            patience: 容忍多少个 epoch 没有显著提升
            min_delta: 最小提升幅度（AP 提升小于此值视为无提升）
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0
    
    def step(self, score, epoch):
        """
        更新 early stopping 状态
        
        Args:
            score: 当前 epoch 的验证指标（AP）
            epoch: 当前 epoch
        
        Returns:
            is_best: 是否是最佳模型
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True
        
        # 检查是否有显著提升
        if self.mode == 'max':
            delta = score - self.best_score
        else:
            delta = self.best_score - score
        
        if delta > self.min_delta:
            # 有显著提升
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            # 无显著提升
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
    
    def __str__(self):
        return (f"EarlyStopping(best={self.best_score:.4f}, "
                f"best_epoch={self.best_epoch}, "
                f"counter={self.counter}/{self.patience})")


def extract_and_save_prototypes(model, dataloader, device, save_path, max_batches=None):
    """
    提取并保存类别原型特征
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        save_path: 保存路径
        max_batches: 最多处理多少个 batch（None 表示全部）
    """
    print(f"\n{'='*80}")
    print("📊 提取类别原型特征...")
    print(f"{'='*80}\n")
    
    extractor = FeatureExtractor(model, device)
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        if batch_idx % 10 == 0:
            print(f"处理 batch {batch_idx}/{len(dataloader)}...")
        
        features_dict = extractor.extract_features_from_batch(images, targets)
        extractor.accumulate_features(features_dict)
    
    # 计算原型
    print("\n计算类别原型...")
    prototypes, stats = extractor.compute_class_prototypes()
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print("类别原型统计：")
    print(f"{'='*80}")
    print(f"{'Class ID':<10} {'样本数':<10} {'特征维度':<10} {'特征分散度':<15}")
    print(f"{'-'*80}")
    
    from visualize_ssod_data import COCO_CLASSES
    for class_id in sorted(prototypes.keys()):
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
        proto = prototypes[class_id]
        stat = stats[class_id]
        print(f"{class_id:<10} {stat['count']:<10} {proto.shape[0]:<10} {stat['std']:<15.4f}  ({class_name})")
    
    print(f"{'-'*80}")
    print(f"总计: {len(prototypes)} 个类别\n")
    
    # 保存
    save_dict = {
        'prototypes': prototypes,
        'stats': stats,
        'class_count': len(prototypes),
        'feature_dim': next(iter(prototypes.values())).shape[0] if prototypes else 0,
    }
    
    torch.save(save_dict, save_path)
    print(f"✅ 类别原型已保存到: {save_path}\n")
    
    return prototypes, stats


def train_with_early_stopping(args):
    """
    带 Early Stopping 的训练主函数
    """
    # 设置分布式
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    
    # 加载配置
    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() 
                       if k not in ['update'] and v is not None})
    
    cfg = YAMLConfig(args.config, **update_dict)
    
    print(f"\n{'='*80}")
    print("🎯 监督预训练 - Encoder 特征学习")
    print(f"{'='*80}")
    print(f"配置文件: {args.config}")
    print(f"输出目录: {cfg.output_dir}")
    print(f"总 Epoch 数: {args.max_epochs}")
    print(f"Early Stopping 耐心值: {args.patience}")
    print(f"最小 AP 提升: {args.min_delta}")
    print(f"{'='*80}\n")
    
    # 创建 solver（使用标准的目标检测训练）
    # 注意：这里使用 'detection' 任务而非 'ssod'
    task_name = 'detection'  # 纯监督训练
    if task_name not in TASKS:
        print(f"⚠️  Warning: Task '{task_name}' not found, using cfg.yaml_cfg['task']")
        task_name = cfg.yaml_cfg.get('task', 'detection')
    
    solver = TASKS[task_name](cfg)
    
    # 初始化 solver（创建 dataloader、optimizer 等）
    solver.train()
    
    # 加载预训练模型权重（如果指定）
    if hasattr(args, 'pretrained') and args.pretrained:
        print(f"\n{'='*80}")
        print("📦 加载预训练模型...")
        print(f"{'='*80}")
        print(f"预训练模型路径: {args.pretrained}")
        
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        
        # 提取模型权重
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"✅ 从检查点加载模型权重（Epoch {checkpoint.get('epoch', 'N/A')}）")
            if 'best_ap' in checkpoint or 'ap' in checkpoint:
                ap_value = checkpoint.get('best_ap', checkpoint.get('ap', 0))
                print(f"   预训练模型 AP: {ap_value:.4f}")
        elif 'ema' in checkpoint:
            state_dict = checkpoint['ema']['module']
            print(f"✅ 从检查点加载 EMA 模型权重")
        else:
            state_dict = checkpoint
            print(f"✅ 直接加载模型权重")
        
        # 处理 DDP 前缀（双向处理：去除或添加）
        # 首先检查当前模型是否需要 'module.' 前缀
        model_keys = set(solver.model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # 判断是否需要转换
        model_has_module = any(k.startswith('module.') for k in model_keys)
        ckpt_has_module = any(k.startswith('module.') for k in ckpt_keys)
        
        new_state_dict = {}
        if ckpt_has_module and not model_has_module:
            # 检查点有前缀，模型没有 -> 去除前缀
            print("   检测到 DDP 前缀不匹配，正在去除 'module.' 前缀...")
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
        elif not ckpt_has_module and model_has_module:
            # 检查点没有前缀，模型有 -> 添加前缀
            print("   检测到 DDP 前缀不匹配，正在添加 'module.' 前缀...")
            for k, v in state_dict.items():
                new_state_dict[f'module.{k}'] = v
        else:
            # 前缀一致，直接使用
            new_state_dict = state_dict
        
        # 加载权重
        missing_keys, unexpected_keys = solver.model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        
        print(f"✅ 预训练模型加载完成")
        
        # 冻结特定模块（如果指定）
        if hasattr(args, 'freeze_backbone') and args.freeze_backbone:
            print("🔒 冻结 Backbone...")
            if hasattr(solver.model, 'backbone'):
                for param in solver.model.backbone.parameters():
                    param.requires_grad = False
                print("✅ Backbone 已冻结")
        
        if hasattr(args, 'freeze_encoder') and args.freeze_encoder:
            print("🔒 冻结 Encoder...")
            if hasattr(solver.model, 'encoder'):
                for param in solver.model.encoder.parameters():
                    param.requires_grad = False
                print("✅ Encoder 已冻结")
        
        # 打印可训练参数统计
        total_params = sum(p.numel() for p in solver.model.parameters())
        trainable_params = sum(p.numel() for p in solver.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"\n📊 参数统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"   冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        print(f"{'='*80}\n")
    
    # 确保 solver 有必要的属性（DetSolver.fit() 中会设置这些）
    if not hasattr(solver, 'self_lr_scheduler'):
        solver.self_lr_scheduler = False
        if cfg.lrsheduler is not None:
            from engine.optim.lr_scheduler import FlatCosineLRScheduler
            iter_per_epoch = len(solver.train_dataloader)
            solver.lr_scheduler = FlatCosineLRScheduler(
                solver.optimizer, cfg.lr_gamma, iter_per_epoch,
                total_epochs=cfg.epoches,
                warmup_iter=cfg.warmup_iter,
                flat_epochs=cfg.flat_epoch,
                no_aug_epochs=cfg.no_aug_epoch
            )
            solver.self_lr_scheduler = True
    
    # 初始化 Early Stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='max'
    )
    
    # 如果使用 --resume，恢复 Early Stopping 的最佳值
    if hasattr(args, 'resume') and args.resume:
        try:
            import torch
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'best_ap' in checkpoint:
                early_stopping.best_score = checkpoint['best_ap']
                print(f"✅ 恢复 Early Stopping 最佳 AP: {checkpoint['best_ap']:.4f}")
            elif 'ap' in checkpoint:
                early_stopping.best_score = checkpoint['ap']
                print(f"✅ 恢复 Early Stopping 最佳 AP: {checkpoint['ap']:.4f}")
        except Exception as e:
            print(f"⚠️  警告：无法从检查点读取最佳 AP: {e}")
    
    # 训练循环
    print(f"🚀 开始训练...\n")
    
    # 获取起始 epoch（兼容不同 solver）
    if hasattr(solver, 'last_epoch'):
        start_epoch = solver.last_epoch + 1
    elif hasattr(solver.cfg, 'last_epoch'):
        start_epoch = solver.cfg.last_epoch + 1
    else:
        start_epoch = 0
    
    # 如果使用了 --resume，确保从正确的 epoch 继续
    if hasattr(args, 'resume') and args.resume and start_epoch == 0:
        # 尝试从检查点读取 epoch
        try:
            import torch
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"✅ 从检查点恢复，将从 Epoch {start_epoch} 继续训练")
            elif 'last_epoch' in checkpoint:
                start_epoch = checkpoint['last_epoch'] + 1
                print(f"✅ 从检查点恢复，将从 Epoch {start_epoch} 继续训练")
        except Exception as e:
            print(f"⚠️  警告：无法从检查点读取 epoch 信息: {e}")
            print(f"   将从 Epoch 0 开始")
    
    print(f"📊 训练范围: Epoch {start_epoch} -> {args.max_epochs - 1}")
    print()
    
    best_model_path = None
    
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.max_epochs - 1}")
        print(f"{'='*80}\n")
        
        # 设置模型为训练模式
        solver.model.train()
        
        # 设置 dataloader epoch（用于分布式训练）
        if hasattr(solver.train_dataloader, 'set_epoch'):
            solver.train_dataloader.set_epoch(epoch)
        if dist_utils.is_dist_available_and_initialized():
            if hasattr(solver.train_dataloader, 'sampler') and hasattr(solver.train_dataloader.sampler, 'set_epoch'):
                solver.train_dataloader.sampler.set_epoch(epoch)
        
        # 训练一个 epoch
        train_stats = train_one_epoch(
            solver.self_lr_scheduler if hasattr(solver, 'self_lr_scheduler') else False,
            solver.lr_scheduler,
            solver.model,
            solver.criterion,
            solver.train_dataloader,
            solver.optimizer,
            solver.device,
            epoch,
            max_norm=cfg.clip_max_norm if hasattr(cfg, 'clip_max_norm') else 0,
            print_freq=cfg.print_freq if hasattr(cfg, 'print_freq') else 50,
            ema=solver.ema,
            scaler=solver.scaler,
            lr_warmup_scheduler=solver.lr_warmup_scheduler if hasattr(solver, 'lr_warmup_scheduler') else None,
            writer=solver.writer if hasattr(solver, 'writer') else None
        )
        
        # 验证
        solver.model.eval()
        module = solver.ema.module if solver.ema else solver.model
        test_stats, coco_evaluator = evaluate(
            module,
            solver.criterion,
            solver.postprocessor,
            solver.val_dataloader,
            solver.evaluator,
            solver.device
        )
        
        # 获取 AP
        if test_stats and 'coco_eval_bbox' in test_stats:
            ap = test_stats['coco_eval_bbox'][0]  # AP@0.50:0.95
        else:
            print("⚠️  Warning: 无法获取 AP，跳过 early stopping 检查")
            ap = 0.0
        
        # Early Stopping 检查
        is_best = early_stopping.step(ap, epoch)
        
        print(f"\n📊 当前 AP: {ap:.4f}")
        print(f"📊 最佳 AP: {early_stopping.best_score:.4f} (Epoch {early_stopping.best_epoch})")
        print(f"📊 {early_stopping}")
        
        # 保存检查点
        if dist_utils.is_main_process():
            checkpoint = {
                'model': solver.model.state_dict() if not hasattr(solver.model, 'module') else solver.model.module.state_dict(),
                'optimizer': solver.optimizer.state_dict(),
                'epoch': epoch,
                'ap': ap,
                'best_ap': early_stopping.best_score,
            }
            
            if solver.ema is not None:
                checkpoint['ema'] = solver.ema.module.state_dict() if hasattr(solver.ema, 'module') else solver.ema.state_dict()
            
            # 保存最新检查点
            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            latest_path = output_dir / 'checkpoint_latest.pth'
            torch.save(checkpoint, latest_path)
            print(f"✅ 保存最新检查点: {latest_path}")
            
            # 如果是最佳模型，额外保存
            if is_best:
                best_model_path = output_dir / f'checkpoint_best_ap{ap:.4f}_epoch{epoch:03d}.pth'
                torch.save(checkpoint, best_model_path)
                print(f"🏆 保存最佳检查点: {best_model_path}")
        
        # 检查是否应该停止
        if early_stopping.should_stop:
            print(f"\n{'='*80}")
            print(f"🛑 Early Stopping 触发！")
            print(f"   - 最佳 AP: {early_stopping.best_score:.4f}")
            print(f"   - 最佳 Epoch: {early_stopping.best_epoch}")
            print(f"   - 当前 Epoch: {epoch}")
            print(f"   - 无显著提升轮数: {early_stopping.counter}")
            print(f"{'='*80}\n")
            break
    
    # 训练结束，提取类别原型
    if dist_utils.is_main_process() and best_model_path is not None:
        print(f"\n{'='*80}")
        print("🎓 训练完成！开始提取类别原型特征...")
        print(f"{'='*80}\n")
        
        # 加载最佳模型
        print(f"📦 加载最佳模型: {best_model_path}")
        state = torch.load(best_model_path, map_location=solver.device)
        
        if hasattr(solver.model, 'module'):
            solver.model.module.load_state_dict(state['model'])
        else:
            solver.model.load_state_dict(state['model'])
        
        # 提取原型
        output_dir = Path(cfg.output_dir)
        prototype_path = output_dir / 'class_prototypes.pth'
        prototypes, stats = extract_and_save_prototypes(
            model=solver.model,
            dataloader=solver.train_dataloader,
            device=solver.device,
            save_path=prototype_path,
            max_batches=args.max_prototype_batches
        )
        
        # 保存完整的预训练结果
        final_save_path = output_dir / 'supervised_pretrain_final.pth'
        final_checkpoint = {
            'model': state['model'],
            'epoch': state['epoch'],
            'ap': state['ap'],
            'prototypes': prototypes,
            'stats': stats,
        }
        if 'ema' in state:
            final_checkpoint['ema'] = state['ema']
        
        torch.save(final_checkpoint, final_save_path)
        print(f"✅ 最终预训练模型已保存: {final_save_path}")
        
        print(f"\n{'='*80}")
        print("📋 预训练总结：")
        print(f"{'='*80}")
        print(f"  - 最佳 AP: {early_stopping.best_score:.4f}")
        print(f"  - 最佳 Epoch: {early_stopping.best_epoch}")
        print(f"  - 总训练轮数: {epoch + 1}")
        print(f"  - 提取类别数: {len(prototypes)}")
        print(f"  - 模型保存: {final_save_path}")
        print(f"  - 原型保存: {prototype_path}")
        print(f"{'='*80}\n")
    
    dist_utils.cleanup()


def _worker(local_rank: int, args):
    """多 GPU 训练的 worker 函数"""
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29600')  # 使用不同端口避免冲突
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(args.num_gpus)
    args.local_rank = local_rank
    train_with_early_stopping(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='监督预训练 - Encoder 特征学习')
    
    # 基础配置
    parser.add_argument('-c', '--config', type=str, 
                       default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml',
                       help='配置文件路径')
    parser.add_argument('-r', '--resume', type=str, default='outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth', help='从检查点恢复（包括训练状态）')
    parser.add_argument('--pretrained', type=str,
                       help='加载预训练模型权重继续训练（只加载模型权重，不加载训练状态）')
    parser.add_argument('-d', '--device', type=str, default=None, help='设备')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--use-amp', action='store_true', default=True, help='使用混合精度')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    # 模型冻结选项（与 --pretrained 配合使用）
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='冻结 Backbone（不更新其参数）')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='冻结 Encoder（不更新其参数）')
    
    # 训练配置
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='最大训练轮数（默认 100）')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping 耐心值（默认 10）')
    parser.add_argument('--min-delta', type=float, default=0.001,
                       help='最小 AP 提升幅度（默认 0.001）')
    
    # 特征提取配置
    parser.add_argument('--max-prototype-batches', type=int, default=500,
                       help='提取原型时最多处理多少个 batch（默认 500，None 表示全部）')
    
    # 分布式配置
    parser.add_argument('--num-gpus', type=int, default=2, help='GPU 数量')
    parser.add_argument('--gpu-ids', type=str, default='1,2', help='GPU IDs')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:29600')
    
    # 其他
    parser.add_argument('-u', '--update', nargs='+', help='更新配置')
    parser.add_argument('--print-method', type=str, default='builtin')
    parser.add_argument('--print-rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    
    args = parser.parse_args()
    
    # 设置可见 GPU
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    # 多 GPU 训练
    if args.num_gpus and args.num_gpus > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        mp.spawn(_worker, nprocs=args.num_gpus, args=(args,), join=True)
    else:
        train_with_early_stopping(args)

