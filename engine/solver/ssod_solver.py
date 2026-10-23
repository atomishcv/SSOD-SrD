"""
Semi-DETR 风格的半监督训练 Solver

核心设计：
1. 使用双 DataLoader 模式（稳定且已验证）
2. 有标签数据使用标准 DataLoader
3. 无标签数据使用 MultiBranchCollateFunction 生成 weak/strong 两个分支
4. 保留 HSV 支持
"""

import json
import os
import time
import datetime
from pathlib import Path
from typing import Optional

import torch

from ._solver import BaseSolver
from .det_engine import evaluate_merged_classes
from .ssod_engine import train_one_epoch_ssod, save_teacher_box_scores
from .semi_detr_modules import AdaptiveThresholdManager
from ..misc import dist_utils
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ..data.dataset.semi_dataset import SemiDataLoaderWrapper


class SSODSolver(BaseSolver):
    """
    Semi-DETR 风格的半监督训练 Solver
    
    使用双 DataLoader 模式：一个用于有标签数据，一个用于无标签数据
    """

    def fit(self):
        """Semi-DETR 风格的训练循环"""
        # 先调用父类的 train() 初始化模型等
        self.train()
        args = self.cfg
        
        # 🔥 检查并获取 unlabeled_dataloader（单卡模式直接使用配置中的 loader）
        if hasattr(self.cfg, 'unlabeled_dataloader') and self.cfg.unlabeled_dataloader is not None:
            # 单卡模式：直接使用配置中的 dataloader（warp_loader 在单卡模式下不做任何修改）
            self.unlabeled_dataloader = self.cfg.unlabeled_dataloader
            if dist_utils.is_main_process():
                print(f"✅ unlabeled_dataloader 已加载")
                print(f"   有标签数据: {len(self.train_dataloader.dataset)} 张")
                print(f"   无标签数据: {len(self.unlabeled_dataloader.dataset)} 张")
        else:
            raise RuntimeError("SSOD 训练需要 unlabeled_dataloader 配置")
        
        # 🔥 使用 SemiDataLoaderWrapper 合并双 DataLoader（解决 iterator 同步问题）
        # 从配置文件读取 sample_ratio
        sample_ratio = args.yaml_cfg.get('ssl_sample_ratio', [1, 4])
        self.semi_dataloader = SemiDataLoaderWrapper(
            self.train_dataloader,
            self.unlabeled_dataloader,
            sample_ratio=sample_ratio  # 🔥 从配置文件读取采样比例
        )
        if dist_utils.is_main_process():
            print(f"🔥 使用 SemiDataLoaderWrapper 模式（sample_ratio={sample_ratio}）")
            print(f"   每个 epoch 的 iteration 数: {len(self.semi_dataloader)}")
        
        # 构建学习率调度器（使用 semi_dataloader 的长度）
        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.semi_dataloader)  # 🔥 使用合并后的 DataLoader 长度
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer, args.lr_gamma, iter_per_epoch, 
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter, 
                flat_epochs=args.flat_epoch, 
                no_aug_epochs=args.no_aug_epoch
            )
            self.self_lr_scheduler = True

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        
        # 🎯 自适应阈值参数（默认值与配置文件对齐）
        use_adaptive_threshold = args.yaml_cfg.get('ssl_use_adaptive_threshold', False)
        adaptive_thresh_quantile = args.yaml_cfg.get('ssl_adaptive_quantile', 0.95)
        adaptive_thresh_min = args.yaml_cfg.get('ssl_adaptive_tau_min', 0.10)  # 与配置文件对齐
        adaptive_thresh_max = args.yaml_cfg.get('ssl_adaptive_tau_max', 0.35)  # 与配置文件对齐
        
        # 类别名称（用于日志，兼容性保留）
        class_names = args.yaml_cfg.get('ssl_class_names', ['small', 'big'])
        num_classes = len(class_names)
        
        # 🎯 初始化自适应阈值管理器（存储为实例属性，以便自动保存到 checkpoint）
        if use_adaptive_threshold:
            self.adaptive_threshold_manager = AdaptiveThresholdManager(
                num_classes=num_classes,  # 保留兼容性
                min_samples=50,
                quantile=adaptive_thresh_quantile,  # 🔥 95%分位数
                tau_min=adaptive_thresh_min,
                tau_max=adaptive_thresh_max,
                fallback_quantile=0.7,
                class_names=class_names,  # 保留兼容性
                device=self.device,
            )
            if dist_utils.is_main_process():
                print(f"🎯 启用自适应阈值（每张图片独立计算）: quantile={adaptive_thresh_quantile}, "
                      f"range=[{adaptive_thresh_min}, {adaptive_thresh_max}]")
            
            # 🔥 加载挂起的自适应阈值状态（从 checkpoint 恢复）
            if hasattr(self, '_pending_adaptive_state') and self._pending_adaptive_state is not None:
                self.adaptive_threshold_manager.load_state_dict(self._pending_adaptive_state)
                if dist_utils.is_main_process():
                    print(f"📂 恢复自适应阈值状态: {self.adaptive_threshold_manager.get_status_string()}")
                self._pending_adaptive_state = None
        else:
            self.adaptive_threshold_manager = None
        
        
        if dist_utils.is_main_process():
            # 🔥 从配置读取参数用于日志打印
            nms_iou_thresh = args.yaml_cfg.get('ssl_pseudo_nms_iou', 0.5)
            conf_thresh = args.yaml_cfg.get('conf_thresh', 0.3)
            if use_adaptive_threshold:
                print(f"📋 伪标签过滤策略: 自适应阈值(GMM) + NMS(IoU={nms_iou_thresh:.2f})")
            else:
                print(f"📋 伪标签过滤策略: 固定阈值≥{conf_thresh:.2f} + NMS(IoU={nms_iou_thresh:.2f})")
        
        
        # 基于 Epoch 的训练
        ssl_total_epochs = args.yaml_cfg.get('ssl_total_epochs', None)
        if ssl_total_epochs is not None and dist_utils.is_main_process():
            print(f"📊 基于 Epoch 的训练模式：目标总 epoch = {ssl_total_epochs}")

        # 🔥 教师框得分保存配置（每 N 个 epoch 保存 30 张影像的 300 个得分到 score_records/）
        save_score = args.yaml_cfg.get('ssl_save_score_data', False)
        score_interval = args.yaml_cfg.get('ssl_save_score_interval', 10)
        score_image_list = args.yaml_cfg.get('ssl_score_image_list', '/data/DEIMv2/selected_images_30.txt')
        if dist_utils.is_main_process():
            print(f"📊 教师框得分保存: ssl_save_score_data={save_score}, interval={score_interval}, "
                  f"image_list={score_image_list}, output={self.output_dir}")
        
        # 🔥 训练循环
        for epoch in range(start_epoch, args.epoches):
            # 设置 epoch 用于采样器（SemiDataLoaderWrapper 会自动转发）
            self.semi_dataloader.set_epoch(epoch)
            
            # 🔥 使用 SemiDataLoaderWrapper 单 DataLoader 模式（解决同步问题）
            train_stats = train_one_epoch_ssod(
                args,
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.semi_dataloader,        # 🔥 合并后的 DataLoader
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                # 🔥 以下参数现在从 cfg 自动读取，只需传递必要的对象引用
                adaptive_threshold_manager=self.adaptive_threshold_manager,
                visualize_class_names=class_names,  # 类别名称需要传递
            )
            
            # 🔥 每 10 个 epoch 保存教师模型对指定影像的 300 个目标框得分（不做绘图）
            if dist_utils.is_main_process() and save_score and epoch % score_interval == 0 and self.output_dir:
                eval_model = self.ema.module if self.ema is not None else self.model
                save_teacher_box_scores(
                    eval_model,
                    score_image_list,
                    self.device,
                    epoch,
                    Path(self.output_dir),
                    batch_size=8,
                )
            
            # 检查是否达到目标总 epoch 数
            if ssl_total_epochs is not None and epoch + 1 >= ssl_total_epochs:
                    if dist_utils.is_main_process():
                        print(f"✅ 已达到目标总 epoch 数 ({ssl_total_epochs})，训练完成")
                    break

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            
            # 🎯 打印自适应阈值状态（每个 epoch 结束时）
            if use_adaptive_threshold and self.adaptive_threshold_manager is not None:
                if dist_utils.is_main_process():
                    thresh_status = self.adaptive_threshold_manager.get_status_string()
                    print(f"🎯 自适应阈值状态 (Epoch {epoch}): {thresh_status}")

            eval_model = self.ema.module if self.ema is not None else self.model

            # 合并类别评估（small和big合并为单一类别，已包含嵌套框过滤）
            merge_nms_iou = args.yaml_cfg.get('eval_merge_nms_iou', 0.6)  # 默认从 0.5 提高到 0.6
            merge_containment = args.yaml_cfg.get('eval_merge_containment', 0.8)  # 包含度阈值
            test_stats, coco_evaluator = evaluate_merged_classes(
                eval_model,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator.coco_gt,
                self.device,
                nms_iou_threshold=merge_nms_iou,
                containment_threshold=merge_containment
            )
            
            # 打印评估结果
            if dist_utils.is_main_process():
                print("\n" + "="*60)
                print("📊 评估结果（Epoch {}）:".format(epoch))
                if 'coco_eval_bbox' in test_stats:
                    ap = test_stats['coco_eval_bbox'][0]
                    ap50 = test_stats['coco_eval_bbox'][1] if len(test_stats['coco_eval_bbox']) > 1 else 0
                    print(f"   🎯 合并类别评估 (模型 small+big → 单类别 object)")
                    print(f"   📍 已去除重复框 (NMS IoU={merge_nms_iou}, 嵌套阈值={merge_containment})")
                    print(f"   ✅ AP = {ap:.4f}, AP50 = {ap50:.4f}")
                print("="*60 + "\n")

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
            }
            
            if 'test_coco_eval_bbox' in log_stats and isinstance(log_stats['test_coco_eval_bbox'], list):
                log_stats['test_coco_eval_bbox'] = log_stats['test_coco_eval_bbox'][:2]
            
            if use_adaptive_threshold:
                if 'train_adaptive_thresh' in train_stats:
                    log_stats['adaptive_thresh'] = train_stats['train_adaptive_thresh']
                    log_stats['adaptive_samples'] = train_stats.get('train_adaptive_samples', 0)
                elif self.adaptive_threshold_manager is not None:
                    log_stats['adaptive_thresh'] = self.adaptive_threshold_manager.threshold
                    log_stats['adaptive_samples'] = len(self.adaptive_threshold_manager.score_buffer)

            # 保存模型检查点
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint_latest.pth']
                
                # 🔥 使用合并单类别评估的 AP 作为保存最佳模型的标准
                if 'coco_eval_bbox' in test_stats and len(test_stats['coco_eval_bbox']) > 0:
                    current_ap = test_stats['coco_eval_bbox'][0]  # 合并单类别评估的 AP
                    
                    best_ap_file = None
                    best_ap = 0.0
                    
                    if self.output_dir.exists():
                        for f in self.output_dir.glob('checkpoint_best_ap*.pth'):
                            try:
                                import re
                                m = re.search(r'checkpoint_best_ap([\d.]+)_epoch', f.name)
                                if m:
                                    ap_val = float(m.group(1))
                                    if ap_val > best_ap:
                                        best_ap = ap_val
                                        best_ap_file = f
                            except:
                                continue
                    
                    if current_ap > best_ap:
                        new_best_path = self.output_dir / f'checkpoint_best_ap{current_ap:.4f}_epoch{epoch:03d}.pth'
                        checkpoint_paths.append(new_best_path)
                        
                        if best_ap_file and best_ap_file.exists():
                            if dist_utils.is_main_process():
                                print(f"🔄 更新最佳模型（合并单类别）: AP {best_ap:.4f} → {current_ap:.4f}")
                                try:
                                    best_ap_file.unlink()
                                except:
                                    pass
                
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)
                    if dist_utils.is_main_process() and 'best' in str(checkpoint_path):
                        print(f"✅ 保存最佳模型（合并单类别 AP={current_ap:.4f}）: {checkpoint_path.name}")

            # 保存日志
            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def state_dict(self):
        """
        重写 state_dict，添加自适应阈值管理器和原型状态的保存
        """
        # 调用父类的 state_dict
        state = super().state_dict()
        
        # 🎯 保存自适应阈值管理器状态
        if hasattr(self, 'adaptive_threshold_manager') and self.adaptive_threshold_manager is not None:
            state['adaptive_threshold_manager'] = self.adaptive_threshold_manager.state_dict()
            if dist_utils.is_main_process():
                print(f"💾 保存自适应阈值状态: {self.adaptive_threshold_manager.get_status_string()}")
        
        # 🔥 保存语义原型队列状态
        from .semi_detr_modules import get_prototype_state_dict
        proto_state = get_prototype_state_dict()
        if proto_state is not None:
            state['semantic_prototype_queue'] = proto_state
            if dist_utils.is_main_process():
                stats = proto_state.get('total_added', {})
                queue_lens = {k: len(v) if hasattr(v, '__len__') else 0 for k, v in proto_state.get('queues', {}).items()}
                print(f"💾 保存语义原型队列: 队列长度={queue_lens}, 累计添加={stats}")
        
        return state
    
    def load_state_dict(self, state):
        """
        重写 load_state_dict，添加自适应阈值管理器和原型状态的恢复
        """
        # 调用父类的 load_state_dict
        super().load_state_dict(state)
        
        # 🎯 恢复自适应阈值管理器状态
        if 'adaptive_threshold_manager' in state:
            if hasattr(self, 'adaptive_threshold_manager') and self.adaptive_threshold_manager is not None:
                self.adaptive_threshold_manager.load_state_dict(state['adaptive_threshold_manager'])
                if dist_utils.is_main_process():
                    print(f"📂 恢复自适应阈值状态: {self.adaptive_threshold_manager.get_status_string()}")
            else:
                # 🔥 保存到临时变量，等 __init__ 完成后再加载
                self._pending_adaptive_state = state['adaptive_threshold_manager']
                if dist_utils.is_main_process():
                    print("📦 保存自适应阈值状态供稍后加载...")
        else:
            if hasattr(self, 'adaptive_threshold_manager') and self.adaptive_threshold_manager is not None:
                if dist_utils.is_main_process():
                    print("⚠️ checkpoint 中没有自适应阈值状态，使用初始值")
        
        # 🔥 恢复语义原型队列状态
        if 'semantic_prototype_queue' in state:
            from .semi_detr_modules import load_prototype_state_dict
            load_prototype_state_dict(state['semantic_prototype_queue'])
            if dist_utils.is_main_process():
                stats = state['semantic_prototype_queue'].get('total_added', {})
                print(f"📂 恢复语义原型队列: 累计添加={stats}")
        