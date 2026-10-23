"""
半监督训练 Solver（参考 Semi-DETR）

负责：
1. 构建模型、数据集、优化器
2. 加载预训练权重
3. 训练循环
4. 模型保存和恢复
"""

import os
import copy
import time
import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .deim_ssod import DEIMSemiDetector
from .mean_teacher import MeanTeacher
from .ssod_engine import train_one_epoch_ssod, evaluate, is_main_process
from .ssod_dataset import (
    CocoLabeledDataset, 
    CocoUnlabeledDataset, 
    MultiBranchPipeline,
)


class SSODSolver:
    """
    半监督训练 Solver（参考 Semi-DETR）
    
    使用 DEIMv2 原有的配置系统构建模型
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg: YAMLConfig 配置对象
        """
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从 yaml_cfg 获取配置
        self.yaml_cfg = cfg.yaml_cfg
        
        # 构建模型（使用 DEIMv2 原有的配置系统）
        self.model = self._build_model()
        
        # 使用 DEIMv2 原有的 criterion
        self.criterion = cfg.criterion
        
        # 构建数据集
        self.train_loader, self.unlabeled_loader = self._build_dataloaders()
        self.val_loader = self._build_val_dataloader()
        
        # 构建优化器
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        
        # EMA 教师
        ema_momentum = self.yaml_cfg.get('ema_momentum', 0.999)
        ema_warmup = self.yaml_cfg.get('ema_warmup', 100)
        
        self.mean_teacher = MeanTeacher(
            teacher=self.model.teacher,
            student=self.model.student,
            momentum=ema_momentum,
            warm_up=ema_warmup,
        )
        
        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if self.yaml_cfg.get('use_amp', True) else None
        
        # 输出目录
        output_dir = self.yaml_cfg.get('output_dir', './outputs/deim_ssod')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.start_epoch = 0
        self.best_ap = 0.0
    
    def _build_model(self) -> DEIMSemiDetector:
        """构建半监督检测模型"""
        # 使用 DEIMv2 原有的配置系统构建学生模型
        # cfg.model 会自动调用 create() 函数
        student = self.cfg.model
        student.to(self.device)
        
        # 深拷贝创建教师模型
        teacher = copy.deepcopy(student)
        teacher.to(self.device)
        
        # 获取配置参数
        hidden_dim = self.yaml_cfg.get('hidden_dim', 256)
        num_classes = self.yaml_cfg.get('num_classes', 80)
        unsup_weight = self.yaml_cfg.get('unsup_weight', 4.0)
        pseudo_label_score_thr = self.yaml_cfg.get('pseudo_label_score_thr', 0.5)
        
        # 构建半监督检测器
        model = DEIMSemiDetector(
            student=student,
            teacher=teacher,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            unsup_weight=unsup_weight,
            pseudo_label_score_thr=pseudo_label_score_thr,
        )
        
        return model
    
    def _build_dataloaders(self):
        """构建训练数据加载器"""
        yaml_cfg = self.yaml_cfg
        
        # 获取数据路径
        labeled_img_folder = yaml_cfg.get('labeled_img_folder')
        labeled_ann_file = yaml_cfg.get('labeled_ann_file')
        unlabeled_img_folder = yaml_cfg.get('unlabeled_img_folder')
        unlabeled_ann_file = yaml_cfg.get('unlabeled_ann_file')
        
        img_size = tuple(yaml_cfg.get('img_size', [800, 800]))
        
        # 有标签数据集
        labeled_dataset = CocoLabeledDataset(
            img_folder=labeled_img_folder,
            ann_file=labeled_ann_file,
            size=img_size,
        )
        
        # 无标签数据集
        unlabeled_dataset = CocoUnlabeledDataset(
            img_folder=unlabeled_img_folder,
            ann_file=unlabeled_ann_file,
            size=img_size,
        )
        
        batch_size = yaml_cfg.get('batch_size', 2)
        batch_size_unlabeled = yaml_cfg.get('batch_size_unlabeled', 4)
        num_workers = yaml_cfg.get('num_workers', 4)
        
        # 有标签数据加载器
        labeled_loader = DataLoader(
            labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._labeled_collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        
        # 无标签数据加载器
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size_unlabeled,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self._unlabeled_collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        
        if is_main_process():
            print(f"📊 数据集统计:")
            print(f"   有标签数据: {len(labeled_dataset)} 张")
            print(f"   无标签数据: {len(unlabeled_dataset)} 张")
            print(f"   Batch size: {batch_size} (labeled) / {batch_size_unlabeled} (unlabeled)")
        
        return labeled_loader, unlabeled_loader
    
    def _build_val_dataloader(self):
        """构建验证数据加载器"""
        yaml_cfg = self.yaml_cfg
        
        val_img_folder = yaml_cfg.get('val_img_folder', yaml_cfg.get('labeled_img_folder'))
        val_ann_file = yaml_cfg.get('val_ann_file')
        img_size = tuple(yaml_cfg.get('img_size', [800, 800]))
        
        val_dataset = CocoLabeledDataset(
            img_folder=val_img_folder,
            ann_file=val_ann_file,
            size=img_size,
        )
        
        batch_size = yaml_cfg.get('batch_size', 2)
        num_workers = yaml_cfg.get('num_workers', 4)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._labeled_collate_fn,
            pin_memory=True,
        )
        
        return val_loader
    
    def _labeled_collate_fn(self, batch):
        """有标签数据 collate 函数"""
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets
    
    def _unlabeled_collate_fn(self, batch):
        """无标签数据 collate 函数"""
        weak_images = torch.stack([item['weak'][0] for item in batch])
        weak_targets = [item['weak'][1] for item in batch]
        strong_images = torch.stack([item['strong'][0] for item in batch])
        strong_targets = [item['strong'][1] for item in batch]
        
        return {
            'weak': (weak_images, weak_targets),
            'strong': (strong_images, strong_targets),
        }
    
    def _build_optimizer(self):
        """构建优化器"""
        yaml_cfg = self.yaml_cfg
        
        lr = yaml_cfg.get('lr', 1e-4)
        weight_decay = yaml_cfg.get('weight_decay', 1e-4)
        
        # 只优化学生模型参数
        params = self.model.student.parameters()
        
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
        
        return optimizer
    
    def _build_lr_scheduler(self):
        """构建学习率调度器"""
        yaml_cfg = self.yaml_cfg
        
        epochs = yaml_cfg.get('epochs', 100)
        min_lr = yaml_cfg.get('min_lr', 1e-6)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=min_lr,
        )
        
        return scheduler
    
    def load_pretrained(self, checkpoint_path: str):
        """加载预训练权重"""
        if is_main_process():
            print(f"📥 加载预训练权重: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型权重
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'ema' in checkpoint:
            state_dict = checkpoint['ema']
        else:
            state_dict = checkpoint
        
        # 加载到学生模型
        missing, unexpected = self.model.student.load_state_dict(state_dict, strict=False)
        
        # 复制到教师模型
        self.model.teacher.load_state_dict(self.model.student.state_dict())
        
        if is_main_process():
            print(f"✅ 预训练权重加载完成")
            if missing:
                print(f"   Missing keys: {len(missing)}")
            if unexpected:
                print(f"   Unexpected keys: {len(unexpected)}")
    
    def resume(self, checkpoint_path: str):
        """恢复训练"""
        if is_main_process():
            print(f"📥 恢复训练: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型权重
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        
        # 加载优化器状态
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加载调度器状态
        if 'lr_scheduler' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        # 加载训练状态
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        
        if 'best_ap' in checkpoint:
            self.best_ap = checkpoint['best_ap']
        
        if is_main_process():
            print(f"✅ 恢复训练完成，从 epoch {self.start_epoch} 开始")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'model': self.model.state_dict(),
            'student': self.model.student.state_dict(),
            'teacher': self.model.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_ap': self.best_ap,
        }
        
        # 保存最新检查点
        latest_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            if is_main_process():
                print(f"✅ 保存最佳模型: {best_path}")
        
        # 每 10 个 epoch 保存一次
        if (epoch + 1) % 10 == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch{epoch:03d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def fit(self):
        """训练模型"""
        yaml_cfg = self.yaml_cfg
        
        start_time = time.time()
        
        epochs = yaml_cfg.get('epochs', 100)
        total_iters = yaml_cfg.get('total_iters', None)
        unsup_weight = yaml_cfg.get('unsup_weight', 4.0)
        warmup_iters = yaml_cfg.get('warmup_iters', 12000)
        pseudo_label_score_thr = yaml_cfg.get('pseudo_label_score_thr', 0.5)
        clip_max_norm = yaml_cfg.get('clip_max_norm', 0.1)
        eval_interval = yaml_cfg.get('eval_interval', 5)
        
        if is_main_process():
            print(f"\n🚀 开始半监督训练 (Semi-DETR 风格)")
            print(f"   Epochs: {epochs}")
            print(f"   Total iters: {total_iters}")
            print(f"   Unsup weight: {unsup_weight}")
            print(f"   Warmup iters: {warmup_iters}")
            print(f"   Pseudo label threshold: {pseudo_label_score_thr}")
            print(f"   Output dir: {self.output_dir}\n")
        
        for epoch in range(self.start_epoch, epochs):
            # 训练一个 epoch
            train_stats = train_one_epoch_ssod(
                model=self.model,
                criterion=self.criterion,
                labeled_loader=self.train_loader,
                unlabeled_loader=self.unlabeled_loader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                mean_teacher=self.mean_teacher,
                max_norm=clip_max_norm,
                scaler=self.scaler,
                unsup_weight=unsup_weight,
                warmup_iters=warmup_iters,
                pseudo_label_score_thr=pseudo_label_score_thr,
                total_iters=total_iters,
            )
            
            # 更新学习率
            self.lr_scheduler.step()
            
            # 评估（使用教师模型）
            is_best = False
            if (epoch + 1) % eval_interval == 0:
                if is_main_process():
                    print(f"\n📊 Epoch {epoch} 评估...")
                
                test_stats = evaluate(
                    model=self.mean_teacher.module,
                    criterion=self.criterion,
                    data_loader=self.val_loader,
                    device=self.device,
                )
                
                # 检查是否是最佳模型
                current_ap = test_stats.get('AP', 0.0)
                is_best = current_ap > self.best_ap
                if is_best:
                    self.best_ap = current_ap
                    if is_main_process():
                        print(f"🎉 新最佳 AP: {self.best_ap:.4f}")
            
            # 保存检查点
            if is_main_process():
                self.save_checkpoint(epoch, is_best)
            
            # 检查是否达到目标迭代数
            if total_iters is not None:
                iter_per_epoch = len(self.train_loader)
                if (epoch + 1) * iter_per_epoch >= total_iters:
                    if is_main_process():
                        print(f"\n✅ 达到目标总迭代数 ({total_iters})，训练完成")
                    break
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        if is_main_process():
            print(f"\n🎉 训练完成！")
            print(f"   总时间: {total_time_str}")
            print(f"   最佳 AP: {self.best_ap:.4f}")
    
    def val(self):
        """验证模型"""
        test_stats = evaluate(
            model=self.mean_teacher.module,
            criterion=self.criterion,
            data_loader=self.val_loader,
            device=self.device,
        )
        
        if is_main_process():
            print(f"📊 验证结果: {test_stats}")
        
        return test_stats
