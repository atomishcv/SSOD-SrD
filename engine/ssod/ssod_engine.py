"""
半监督训练引擎（参考 Semi-DETR）

核心设计：
1. 分别计算有监督损失和无监督损失
2. 总损失 = sup_loss + lambda_u * unsup_loss
3. 统一反向传播
4. EMA 更新教师模型
"""

import sys
import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .mean_teacher import MeanTeacher
from .gmm_filter import GMMFilter


class MetricLogger:
    """简单的指标记录器"""
    
    def __init__(self):
        self.meters = {}
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = []
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].append(v)
    
    def get_avg(self, key):
        if key not in self.meters or len(self.meters[key]) == 0:
            return 0.0
        return sum(self.meters[key]) / len(self.meters[key])
    
    def __str__(self):
        result = []
        for k, v in self.meters.items():
            if len(v) > 0:
                avg = sum(v) / len(v)
                result.append(f"{k}: {avg:.4f}")
        return ", ".join(result)


def is_main_process():
    """检查是否是主进程"""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def weighted_loss(loss_dict: Dict[str, torch.Tensor], weight: float, step: int, warmup: int) -> Dict[str, torch.Tensor]:
    """
    加权损失（参考 Semi-DETR weighted_loss）
    
    Warmup 阶段权重线性增加：weight = weight * (step - 1) / warmup
    """
    if warmup > 0 and step <= warmup:
        weight = weight * (step - 1) / warmup
    
    return {k: v * weight for k, v in loss_dict.items()}


def train_one_epoch_ssod(
    model: nn.Module,
    criterion: nn.Module,
    labeled_loader: Iterable,
    unlabeled_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    mean_teacher: MeanTeacher,
    *,
    max_norm: float = 0.1,
    print_freq: int = 50,
    scaler: Optional[GradScaler] = None,
    # Semi-DETR 超参数
    unsup_weight: float = 4.0,
    warmup_iters: int = 12000,
    pseudo_label_score_thr: float = 0.5,
    total_iters: Optional[int] = None,
) -> Dict:
    """
    半监督训练一个 epoch（参考 Semi-DETR）
    
    Args:
        model: DEIMSemiDetector 模型
        criterion: 损失函数
        labeled_loader: 有标签数据加载器
        unlabeled_loader: 无标签数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        mean_teacher: EMA 教师更新器
        max_norm: 梯度裁剪阈值
        print_freq: 打印频率
        scaler: AMP scaler
        unsup_weight: 无监督损失权重
        warmup_iters: 无监督损失 warmup 迭代数
        pseudo_label_score_thr: 伪标签置信度阈值
        total_iters: 目标总迭代数
        
    Returns:
        stats: 训练统计信息
    """
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger()
    
    # 计算当前 epoch 的起始迭代数
    iter_per_epoch = len(labeled_loader)
    epoch_start_iter = epoch * iter_per_epoch
    
    # 创建无标签数据迭代器
    unlabeled_iter = iter(unlabeled_loader)
    
    # 进度条
    if is_main_process():
        pbar = tqdm(
            enumerate(labeled_loader),
            total=iter_per_epoch,
            desc=f"Epoch {epoch}",
            ncols=120,
        )
    else:
        pbar = enumerate(labeled_loader)
    
    for i, sup_data in pbar:
        current_iter = epoch_start_iter + i
        
        # 检查是否达到目标总迭代数
        if total_iters is not None and current_iter >= total_iters:
            if is_main_process():
                print(f"\n✅ 已达到目标总迭代数 ({total_iters})，训练完成")
            break
        
        # ========== 1. 获取数据 ==========
        # 有标签数据
        sup_images, sup_targets = sup_data
        sup_images = sup_images.to(device)
        sup_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in sup_targets]
        
        # 无标签数据
        try:
            unsup_data = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unsup_data = next(unlabeled_iter)
        
        unsup_weak_images, unsup_weak_targets = unsup_data['weak']
        unsup_strong_images, unsup_strong_targets = unsup_data['strong']
        
        unsup_weak_images = unsup_weak_images.to(device)
        unsup_strong_images = unsup_strong_images.to(device)
        unsup_weak_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in t.items()} for t in unsup_weak_targets]
        unsup_strong_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in t.items()} for t in unsup_strong_targets]
        
        # ========== 2. 计算无监督权重 ==========
        # Semi-DETR 风格：线性 warmup
        step = current_iter + 1  # 1-based
        if warmup_iters > 0 and step <= warmup_iters:
            current_unsup_weight = unsup_weight * (step - 1) / warmup_iters
        else:
            current_unsup_weight = unsup_weight
        
        # ========== 3. 前向传播 ==========
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                loss_dict = model.forward_train(
                    sup_images=sup_images,
                    sup_targets=sup_targets,
                    unsup_images_weak=unsup_weak_images,
                    unsup_images_strong=unsup_strong_images,
                    unsup_targets_weak=unsup_weak_targets,
                    unsup_targets_strong=unsup_strong_targets,
                    criterion=criterion,
                )
        else:
            loss_dict = model.forward_train(
                sup_images=sup_images,
                sup_targets=sup_targets,
                unsup_images_weak=unsup_weak_images,
                unsup_images_strong=unsup_strong_images,
                unsup_targets_weak=unsup_weak_targets,
                unsup_targets_strong=unsup_strong_targets,
                criterion=criterion,
            )
        
        # ========== 4. 计算总损失 ==========
        # 有监督损失
        sup_loss = sum(v for k, v in loss_dict.items() if k.startswith('sup_'))
        
        # 无监督损失（带权重）
        unsup_loss = sum(v for k, v in loss_dict.items() if k.startswith('unsup_'))
        
        total_loss = sup_loss + unsup_loss
        
        # ========== 5. 反向传播 ==========
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        
        # ========== 6. EMA 更新教师模型 ==========
        mean_teacher.update(current_iter)
        
        # ========== 7. 记录统计信息 ==========
        metric_logger.update(
            loss=total_loss.item(),
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item() / max(current_unsup_weight, 1e-6),
            unsup_weight=current_unsup_weight,
            lr=optimizer.param_groups[0]['lr'],
            ema_momentum=mean_teacher.get_momentum(current_iter),
        )
        
        # 更新进度条
        if is_main_process() and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss': f'{total_loss.item():.2f}',
                'sup': f'{sup_loss.item():.2f}',
                'unsup': f'{unsup_loss.item() / max(current_unsup_weight, 1e-6):.2f}',
                'λ_u': f'{current_unsup_weight:.2f}',
            })
    
    if is_main_process():
        print(f"\n📊 Epoch {epoch} 统计: {metric_logger}")
    
    return {k: metric_logger.get_avg(k) for k in metric_logger.meters.keys()}


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    evaluator=None,
) -> Dict:
    """
    评估模型
    """
    model.eval()
    
    metric_logger = MetricLogger()
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating", disable=not is_main_process()):
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            
            if evaluator is not None:
                # 获取预测结果并添加到 evaluator
                pred_logits = outputs['pred_logits']
                pred_boxes = outputs['pred_boxes']
                
                for idx, (logits, boxes, target) in enumerate(zip(pred_logits, pred_boxes, targets)):
                    probs = logits.sigmoid()
                    scores, labels = probs.max(dim=-1)
                    
                    # 过滤低置信度预测
                    keep = scores > 0.3
                    
                    result = {
                        'boxes': boxes[keep],
                        'scores': scores[keep],
                        'labels': labels[keep],
                    }
                    
                    evaluator.update(target['image_id'], result)
    
    if evaluator is not None:
        stats = evaluator.summarize()
        return stats
    
    return {}

