"""
Semi-DETR 风格的半监督训练循环

关键设计（参考 Semi-DETR）：
1. 监督和无监督损失计算分离到独立模块（ssod_losses.py）
2. 分别计算 sup_loss 和 unsup_loss
3. 总损失 = sup_loss + lambda_u * unsup_loss
4. 统一反向传播

🔥 新增 Semi-DETR 核心模块：
1. Stage-wise Hybrid Matching（分阶段混合匹配）
2. Cross-view Query Consistency（跨视图查询一致性）
3. Cost-based Pseudo Label Mining（基于匹配代价的伪标签挖掘）
"""

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA, Warmup

# 🔥 Semi-DETR 风格：导入分离的损失计算模块
from .ssod_losses import (
    compute_sup_loss, 
    weighted_loss
)

# 🔥 Semi-DETR 核心模块
from .semi_detr_modules import (
    compute_semi_detr_unsup_loss,
    AdaptiveThresholdManager,
)



def _to_device_obj(obj, device, non_blocking=False):
    """Safely move tensors/tv_tensors recursively to device.
    
    Args:
        obj: Object to move (dict, list, tensor, etc.)
        device: Target device
        non_blocking: If True, use async transfer (requires pin_memory=True in DataLoader)
    """
    if isinstance(obj, dict):
        return {k: _to_device_obj(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_device_obj(v, device, non_blocking) for v in obj]
    if hasattr(obj, 'to'):
        try:
            return obj.to(device, non_blocking=non_blocking)
        except Exception:
            return obj
    return obj


def load_student_and_teacher(
                               model: torch.nn.Module,
                             checkpoint_path: Path,
                               device: torch.device,
    ema: Optional[ModelEMA] = None,
    ema_decay: float = 0.999
):
    """
    加载学生与教师初始状态
    """
    def _align_prefix(state_dict, target_has_module_prefix: bool):
        if not state_dict:
            return state_dict
        sd_has_prefix = next(iter(state_dict)).startswith("module.")
        if sd_has_prefix != target_has_module_prefix:
            state_dict = {
                (k[7:] if sd_has_prefix else f"module.{k}"): v
                for k, v in state_dict.items()
            }
        return state_dict

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in ckpt:
        raise RuntimeError(f"checkpoint 中不含 'model' 权重：{checkpoint_path}")
    state_dict_model = ckpt["model"]

    student_model = model
    student_module = dist_utils.de_parallel(student_model)
    student_module.to(device)
    sd_student = _align_prefix(
        state_dict_model,
        next(iter(student_module.state_dict())).startswith("module.")
    )
    
    student_module.load_state_dict(sd_student, strict=False)
    
    # 🔥 重新创建 EMA 模型（确保使用最新的学生模型权重）
    # 注意：SSOD 训练使用 Semi-DETR 风格的外部动量计算（ssod_engine.py 中的 ema_decay），
    # 因此设置 warmups=0 禁用 ModelEMA 内置的 warmup 逻辑，避免设计不一致
    from copy import deepcopy
    if ema is None:
        ema = ModelEMA(student_module, decay=ema_decay, warmups=0)
    else:
        # 🔥 完全重新创建 EMA 模块，确保所有权重和 buffer 都正确复制
        ema.module = deepcopy(dist_utils.de_parallel(student_module))
        ema.decay = ema_decay
        ema.warmups = 0  # 🔥 禁用内置 warmup，使用外部 Semi-DETR 风格动量计算
        # 重新设置 decay_fn（warmups=0 时为常量函数）
        ema.decay_fn = lambda x: ema_decay
        # 冻结参数
        for p in ema.module.parameters():
            p.requires_grad_(False)

    ema.module.to(device).eval()

    return student_model, ema


def train_one_epoch_ssod(
                        args,
                        self_lr_scheduler: bool,
                         lr_scheduler,
                         model: torch.nn.Module,
                         criterion: torch.nn.Module,
                         labeled_loader: Iterable,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device,
                         epoch: int,
                         max_norm: float = 0,
                         print_freq: int = 10,
                         ema: ModelEMA = None,
                         scaler: GradScaler = None,
                         lr_warmup_scheduler: Warmup = None,
                         # 🔥 以下参数保留用于向后兼容，但优先从 cfg 读取
                         lambda_u: Optional[float] = None,
                         pseudo_thresh: Optional[float] = None,
                         nms_iou: Optional[float] = None,
                        nms_containment: Optional[float] = None,
                        warmup_epochs: Optional[int] = None,
                        ema_momentum: Optional[float] = None,
                        ema_warmup_epochs: Optional[int] = None,
                        ema_burnin_epochs: Optional[int] = None,
                        conf_thresh: Optional[float] = None,
                        consistency_weight: Optional[float] = None,
                        use_hybrid_matching: Optional[bool] = None,
                        hybrid_topk: Optional[int] = None,
                        warm_up_epochs: Optional[int] = None,
                        use_vdvi_filter: Optional[bool] = None,
                        vdvi_thresh: Optional[float] = None,
                        total_epochs: Optional[int] = None,
                        use_semi_detr_dataloader: Optional[bool] = None,
                        visualize_pseudo_labels: Optional[bool] = None,
                        visualize_save_dir: Optional[str] = None,
                        visualize_interval: Optional[int] = None,
                        visualize_class_names: Optional[List[str]] = None,
                        use_adaptive_threshold: Optional[bool] = None,
                        adaptive_threshold_manager: Optional['AdaptiveThresholdManager'] = None,
                                            ) -> dict:
    """
    Semi-DETR 风格的半监督训练循环
    
    🔥 关键设计（参考 Semi-DETR forward_train）：
    
    ```python
    # Semi-DETR 的 forward_train 逻辑：
    loss = {}
    
    if "sup" in data_groups:
        sup_loss = self.student.forward_train(**data_groups["sup"])
        sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
        loss.update(**sup_loss)
    
    if "unsup_student" in data_groups:
        unsup_loss = weighted_loss(
            self.foward_unsup_train(data_groups["unsup_teacher"], data_groups["unsup_student"]),
            weight=self.unsup_weight,
        )
        unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
        loss.update(**unsup_loss)
    
    return loss  # 外部统一反向传播
    ```
    """
    student_model = model
    teacher_model = ema.module if ema is not None else student_model

    student_model.train()
    criterion.train()
    
    # 🔥 QueryProjector 使用全局平均池化（不需要可学习参数）
    # 这样不会影响 checkpoint 的兼容性，且避免优化器参数组数量不匹配的问题
    # 一致性损失主要依赖于 encoder 提取的特征质量，全局平均池化足够使用
    query_projector = None  # 使用 None 表示在 extract_queries_from_pseudo_labels 中使用全局平均池化

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.2e}'))  # 🔥 使用科学计数法显示学习率
    metric_logger.add_meter('ema_momentum', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    if dist_utils.is_main_process():
        print(f"🔥 Semi-DETR modules enabled")
        print(f"📦 使用 SemiDataLoaderWrapper 模式")
    
    # 🔥 SemiDataLoaderWrapper 模式：labeled_loader 实际上是合并后的 semi_dataloader
    semi_dataloader = labeled_loader
    total_iters_in_epoch = len(semi_dataloader)
    
    # 🔥 从配置文件读取超参数（如果参数为 None，则从 cfg 读取）
    cfg = args.yaml_cfg if hasattr(args, 'yaml_cfg') else {}
    
    # 半监督训练参数
    lambda_u = lambda_u if lambda_u is not None else cfg.get('ssl_lambda_u', 2.0)
    pseudo_thresh = pseudo_thresh if pseudo_thresh is not None else cfg.get('ssl_pseudo_score_thresh', 0.0)
    nms_iou = nms_iou if nms_iou is not None else cfg.get('ssl_pseudo_nms_iou', 0.5)
    nms_containment = nms_containment if nms_containment is not None else cfg.get('ssl_pseudo_nms_containment', 0.8)
    warmup_epochs = warmup_epochs if warmup_epochs is not None else cfg.get('ssl_warmup_epochs', 60)
    ema_momentum = ema_momentum if ema_momentum is not None else cfg.get('ssl_ema_momentum', 0.999)
    ema_warmup_epochs = ema_warmup_epochs if ema_warmup_epochs is not None else cfg.get('ssl_ema_warmup_epochs', 3)
    ema_burnin_epochs = ema_burnin_epochs if ema_burnin_epochs is not None else cfg.get('ssl_ema_burnin_epochs', None)
    
    # 🔥 修复：如果 ema_burnin_epochs 为 None，使用 ema_warmup_epochs 的值（此时已从配置读取）
    if ema_burnin_epochs is None:
        ema_burnin_epochs = max(0, ema_warmup_epochs)
    
    conf_thresh = conf_thresh if conf_thresh is not None else cfg.get('conf_thresh', 0.3)
    consistency_weight = consistency_weight if consistency_weight is not None else cfg.get('ssl_consistency_weight', 1.0)
    
    # Semi-DETR 参数
    use_hybrid_matching = use_hybrid_matching if use_hybrid_matching is not None else cfg.get('ssl_use_hybrid_matching', False)
    hybrid_topk = hybrid_topk if hybrid_topk is not None else cfg.get('ssl_hybrid_topk', 13)
    warm_up_epochs = warm_up_epochs if warm_up_epochs is not None else cfg.get('ssl_warm_up_epochs', 100)
    
    # VDVI 参数
    use_vdvi_filter = use_vdvi_filter if use_vdvi_filter is not None else cfg.get('ssl_use_vdvi_filter', True)
    vdvi_thresh = vdvi_thresh if vdvi_thresh is not None else cfg.get('ssl_vdvi_thresh', 0.030)
    
    # 其他参数
    total_epochs = total_epochs if total_epochs is not None else cfg.get('ssl_total_epochs', None)
    use_semi_detr_dataloader = use_semi_detr_dataloader if use_semi_detr_dataloader is not None else cfg.get('use_semi_detr_dataloader', False)
    
    # 可视化参数
    visualize_pseudo_labels = visualize_pseudo_labels if visualize_pseudo_labels is not None else cfg.get('ssl_visualize_pseudo_labels', False)
    # 🔥 修复：处理 output_dir 可能是字符串或 Path 对象的情况
    if visualize_save_dir is None and visualize_pseudo_labels and hasattr(args, 'output_dir') and args.output_dir:
        from pathlib import Path
        output_dir = Path(args.output_dir) if isinstance(args.output_dir, str) else args.output_dir
        visualize_save_dir = str(output_dir / 'pseudo_label_vis')
        # 🔥 确保可视化保存目录存在
        Path(visualize_save_dir).mkdir(parents=True, exist_ok=True)
        if dist_utils.is_main_process():
            print(f"📁 伪标签可视化保存目录: {visualize_save_dir}")
    elif visualize_pseudo_labels and visualize_save_dir is None:
        if dist_utils.is_main_process():
            print(f"⚠️ 警告：可视化已启用但未设置保存目录，将跳过可视化")
    visualize_interval = visualize_interval if visualize_interval is not None else cfg.get('ssl_visualize_interval', 500)
    visualize_class_names = visualize_class_names if visualize_class_names is not None else cfg.get('ssl_class_names', None)
    
    # 自适应阈值参数
    use_adaptive_threshold = use_adaptive_threshold if use_adaptive_threshold is not None else cfg.get('ssl_use_adaptive_threshold', False)
    
    # 🔥 语义原型筛选参数
    use_proto_filter = cfg.get('ssl_use_proto_filter', False)
    proto_filter_start_epoch = cfg.get('ssl_proto_filter_start_epoch', 5)
    if use_proto_filter and dist_utils.is_main_process() and epoch == 0:
        print(f"🔬 语义原型筛选已启用: 从 epoch {proto_filter_start_epoch} 开始")
    
    # 🔥 基于 Epoch 的 warmup 逻辑
    if epoch < warmup_epochs:
        if dist_utils.is_main_process():
            # 计算当前 epoch 的无监督权重比例
            weight_scale = epoch / warmup_epochs
            print(f"📌 Epoch {epoch}: Warmup phase ({epoch}/{warmup_epochs}) - 无监督权重线性增加 (当前={weight_scale:.1%} × {lambda_u:.1f} = {weight_scale * lambda_u:.2f})")
    else:
        if dist_utils.is_main_process():
            print(f"🔥 Epoch {epoch}: Full semi-supervised training - 完整半监督训练 (无监督权重={lambda_u:.1f})")

    # 数据加载器
    data_loader = semi_dataloader
    
    # 🔥 记录 epoch 开始时间
    import time
    epoch_start_time = time.time()
    
    if dist_utils.is_main_process():
        print(f"\n{'='*80}")
        print(f"开始训练 Epoch {epoch} | 总共 {total_iters_in_epoch} iterations")
        print(f"{'='*80}")

    # 🔥 添加 tqdm 进度条（只在主进程显示）
    if dist_utils.is_main_process():
        data_loader_iter = tqdm(
            data_loader,
            desc=f"Epoch {epoch}",
            total=total_iters_in_epoch,
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    else:
        data_loader_iter = data_loader

    for i, batch_data in enumerate(data_loader_iter):
        current_iter = epoch * total_iters_in_epoch + i  # 全局 iteration 计数（用于日志和细粒度控制）
        
        # 🔥 基于 Epoch 的训练：检查是否达到目标总 epoch 数（通常在外层循环检查，这里保留以防万一）
        if total_epochs is not None and epoch >= total_epochs:
            if dist_utils.is_main_process():
                print(f"\n✅ 已达到目标总 epoch 数 ({total_epochs})，提前结束")
            break

        # ========== 1. 获取数据 ==========
        # 🔥 SemiDataLoaderWrapper 模式：直接从字典中获取数据（解决双DataLoader同步问题）
        images_sup, targets_sup = batch_data['sup']
        images_unsup_weak, targets_unsup_weak = batch_data.get('unsup_weak', (None, []))
        images_unsup_strong, targets_unsup_strong = batch_data.get('unsup_strong', (None, []))

        # 🔥 提取几何变换矩阵信息（用于后续坐标对齐）
        if targets_unsup_weak and len(targets_unsup_weak) > 0:
            meta_weak = targets_unsup_weak[0].get('meta', {}) if isinstance(targets_unsup_weak[0], dict) else {}
        else:
            meta_weak = {}
        if targets_unsup_strong and len(targets_unsup_strong) > 0:
            meta_strong = targets_unsup_strong[0].get('meta', {}) if isinstance(targets_unsup_strong[0], dict) else {}
        else:
            meta_strong = {}
        
        transform_weak = {
            'matrix': meta_weak.get('transform_matrix', None),
            'scale': meta_weak.get('scale_factor', None),
            'flip': meta_weak.get('flip', False),
            'translate': meta_weak.get('translate', (0, 0))
        }
        transform_strong = {
            'matrix': meta_strong.get('transform_matrix', None),
            'scale': meta_strong.get('scale_factor', None),
            'flip': meta_strong.get('flip', False),
            'translate': meta_strong.get('translate', (0, 0))
        }

        # 移动到设备
        # 🔥 使用 non_blocking=True 实现异步数据传输（避免 CPU 等待 GPU）
        if images_sup is not None:
            images_sup = {k: v.to(device, non_blocking=True) for k, v in images_sup.items()} if isinstance(images_sup, dict) else images_sup.to(device, non_blocking=True)
        if images_unsup_weak is not None:
            images_unsup_weak = {k: v.to(device, non_blocking=True) for k, v in images_unsup_weak.items()} if isinstance(images_unsup_weak, dict) else images_unsup_weak.to(device, non_blocking=True)
        if images_unsup_strong is not None:
            images_unsup_strong = {k: v.to(device, non_blocking=True) for k, v in images_unsup_strong.items()} if isinstance(images_unsup_strong, dict) else images_unsup_strong.to(device, non_blocking=True)
        
        targets_sup = [_to_device_obj(t, device, non_blocking=True) for t in targets_sup] if targets_sup else []
        targets_unsup_weak = [_to_device_obj(t, device, non_blocking=True) for t in targets_unsup_weak] if targets_unsup_weak else []
        targets_unsup_strong = [_to_device_obj(t, device, non_blocking=True) for t in targets_unsup_strong] if targets_unsup_strong else []
        
        if warmup_epochs == 0:
            # 无 warmup：直接使用完整权重
            unsup_weight_scale = 1.0
        elif epoch < warmup_epochs:
            # Warmup 阶段：线性增加权重
            # weight = weight * epoch / warmup_epochs
            unsup_weight_scale = epoch / warmup_epochs
        else:
            # Warmup 结束后：使用完整权重
            unsup_weight_scale = 1.0
        
        effective_lambda_u = unsup_weight_scale * lambda_u

        if ema is not None:
            teacher_model = ema.module

        metas = dict(epoch=epoch, step=i, global_step=current_iter, epoch_step=total_iters_in_epoch)

        # ========================================================================
        # 🔥 DDP 安全的多次 forward-backward 模式
        # 使用 no_sync() 来避免 "marked as ready twice" 错误
        # ========================================================================
        
        loss = {}  # 汇总所有损失
        consistency_stats = {}
        
        # 🔥 辅助函数：创建安全的零损失（支持 DDP 和 单卡）
        # 🔥 预先缓存一个参数，避免每次都遍历
        _cached_zero_param = next((p for p in student_model.parameters() if p.requires_grad), None)
        
        def create_zero_loss(model):
            """创建一个值为0但有梯度的损失（高效版本，避免遍历参数）"""
            # 🔥 优化：直接使用缓存的参数，避免遍历所有参数导致卡死
            if _cached_zero_param is not None:
                return _cached_zero_param.sum() * 0.0
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 清零梯度
        optimizer.zero_grad(set_to_none=True)
        
        # ========== Step 1: 计算监督损失 ==========
        # 🔥 关键：只有在需要执行无监督分支时才使用 no_sync()
        # 如果 effective_lambda_u == 0（warmup 第一个 iter），不需要无监督分支，直接执行监督分支
        need_unsup_forward = (effective_lambda_u > 0 and images_unsup_weak is not None and len(targets_unsup_weak) > 0)
        
        # 计算监督损失（不立即 backward，等待与无监督损失合并）
        if images_sup is not None and len(targets_sup) > 0:
            # 🎯 修改：compute_sup_loss 现在返回 outputs（包含 encoder_feate）
            loss_sup, loss_dict_sup, sup_outputs = compute_sup_loss(
                student_model=student_model,
                criterion=criterion,
                images=images_sup,
                targets=targets_sup,
                metas=metas,
                scaler=scaler,
                # 🔥 从 cfg 读取参数
                args=args,
                current_epoch=epoch,
            )
            
            # 🔥 监督损失 NaN/Inf 处理：使用 nan_to_num 直接替换，避免 GPU/CPU 同步
            loss_sup = torch.nan_to_num(loss_sup, nan=0.0, posinf=0.0, neginf=0.0)
            
            loss.update(loss_dict_sup)
            
            # 🔥 使用监督数据的 GT 框更新全局类别原型队列（用于伪标签筛选）
            # 特征提取方式与 extract_queries_from_pseudo_labels 一致（多尺度 RoIAlign + FPN 层级选择）
            if use_proto_filter and sup_outputs is not None:
                encoder_feats = sup_outputs.get('encoder_feate', None)
                if encoder_feats is not None:
                    from .semi_detr_modules import update_global_prototypes
                    H_sup, W_sup = images_sup['rgb'].shape[2:]
                    num_classes = cfg.get('num_classes', 2) if args is not None and hasattr(args, 'yaml_cfg') else 2
                    queue_size = cfg.get('ssl_proto_queue_size', 50)  # 🔥 从配置读取队列大小
                    for b_idx, target in enumerate(targets_sup):
                        if 'boxes' in target and 'labels' in target:
                            update_global_prototypes(
                                encoder_feats=encoder_feats,
                                boxes=target['boxes'],
                                labels=target['labels'],
                                batch_idx=b_idx,
                                img_size=(H_sup, W_sup),
                                num_classes=num_classes,
                                queue_size=queue_size,
                            )
            
        else:
            # 没有有标签数据时，设置零损失
            loss_sup = create_zero_loss(student_model)
            loss_dict_sup = {}
        
        # ========== Step 2: Teacher 生成伪标签 ==========
        # 🔥 优化：只在需要无监督损失时才运行 teacher forward
        has_valid_pseudo = False
        num_pseudo_boxes = 0
        teacher_outputs = None
        H_weak, W_weak = 800, 800
        H_strong, W_strong = 800, 800
        
        if need_unsup_forward:
            img_weak_rgb = images_unsup_weak['rgb'] if isinstance(images_unsup_weak, dict) else images_unsup_weak
            img_strong_rgb = images_unsup_strong['rgb'] if isinstance(images_unsup_strong, dict) else images_unsup_strong
            H_weak, W_weak = img_weak_rgb.shape[-2:]
            H_strong, W_strong = img_strong_rgb.shape[-2:]
            
            with torch.no_grad():
                teacher_model.eval()
                images_weak_hsv = images_unsup_weak.get('hsv', None) if isinstance(images_unsup_weak, dict) else None
                
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        teacher_outputs = teacher_model(img_weak_rgb, is_teacher=True, hsv=images_weak_hsv)
                else:
                    teacher_outputs = teacher_model(img_weak_rgb, is_teacher=True, hsv=images_weak_hsv)
        
        # ========== Step 3: 计算无监督损失 ==========
        # 🔥 无监督分支正常执行（不在 no_sync() 中），backward 会触发 all-reduce
        # 同时会同步之前在 no_sync() 中累积的监督分支梯度
        # 🔥 关键：不能根据本地 num_pseudo_boxes 跳过 forward，否则不同 GPU 代码路径不一致会导致 DDP 死锁
        if not need_unsup_forward:
            # 第一个 iteration（step=1 → weight=0）或没有无标签数据：跳过无监督损失计算
            # 🔥 使用所有模型参数创建零损失（确保 DDP 所有参数都收到梯度）
            loss_unsup_weighted = create_zero_loss(student_model)
            loss_unsup = torch.tensor(0.0, device=device)
            loss_dict_unsup = {}
            consistency_stats = {'num_pseudo_boxes': 0, 'num_consistent_queries': 0}
            has_valid_pseudo = False
            proto_filtered = 0
        else:
            # 🌿 提取 VDVI 数据用于伪标签筛选
            vdvi_weak = images_unsup_weak.get('vdvi', None) if isinstance(images_unsup_weak, dict) else None
            
            # 🔥 是否在当前 iteration 进行可视化
            should_visualize = visualize_pseudo_labels and (current_iter % visualize_interval == 0)
            
            loss_unsup_weighted, loss_dict_unsup, consistency_stats = compute_semi_detr_unsup_loss(
                student_model=student_model,
                teacher_model=teacher_model,
                criterion=criterion,
                images_weak=images_unsup_weak,
                images_strong=images_unsup_strong,
                metas=metas,
                # 🔥 从 cfg 读取参数
                args=args,
                unsup_weight=effective_lambda_u,
                curr_epoch=epoch,
                scaler=scaler,
                device=device,
                teacher_outputs=teacher_outputs,
                transform_weak=transform_weak,
                transform_strong=transform_strong,
                img_size_weak=(H_weak, W_weak),
                img_size_strong=(H_strong, W_strong),
                vdvi_weak=vdvi_weak,
                query_projector=query_projector,
                # 🔥 伪标签可视化参数
                visualize_pseudo_labels=should_visualize,
                visualize_save_dir=visualize_save_dir,
                visualize_class_names=visualize_class_names,
                # 🎯 自适应阈值参数
                adaptive_threshold_manager=adaptive_threshold_manager,
            )
            
            # 🔥 这些变量在后续不再使用，会在作用域结束时自动释放
            # 只保留必要的 del（如果变量很大且后续不再使用）
            # del teacher_outputs, images_unsup_weak, images_unsup_strong  # 已优化：减少不必要的释放
            
            num_pseudo_boxes = consistency_stats.get('num_pseudo_boxes', 0)
            proto_filtered = consistency_stats.get('train_proto_filtered', 0)
            
            if 'adaptive_thresh' in consistency_stats:
                metric_logger.update(adaptive_thresh=consistency_stats['adaptive_thresh'])
            if 'adaptive_samples' in consistency_stats:
                metric_logger.update(adaptive_samples=consistency_stats['adaptive_samples'])
            
            # 🔥 双重保险：如果没生成伪标签，强制使用 create_zero_loss 覆盖
            # 确保 DDP 模式下所有参数都参与梯度计算
            if num_pseudo_boxes == 0:
                loss_unsup_weighted = create_zero_loss(student_model)
                loss_dict_unsup = {}
            
            # 🔥 NaN/Inf 检测优化：
            # 1. 使用 nan_to_num 直接替换 NaN/Inf，避免 GPU/CPU 同步
            # 2. 这是纯 GPU 操作，不会触发同步
            loss_unsup_weighted = torch.nan_to_num(loss_unsup_weighted, nan=0.0, posinf=0.0, neginf=0.0)
            
            # loss_unsup 仅用于日志记录（显示未加权的原始损失值）
            if effective_lambda_u > 1e-6:
                loss_unsup = loss_unsup_weighted.detach() / effective_lambda_u
            else:
                loss_unsup = torch.tensor(0.0, device=device)
            
            loss.update(loss_dict_unsup)
            has_valid_pseudo = num_pseudo_boxes > 0
        
        # ========== Step 4: 合并反向传播 ==========
        # 🔥 关键改进：合并监督和无监督损失，统一进行反向传播
        # 这样可以：
        # 1. 简化逻辑，减少 AMP scaler 的错误（只需要一次 backward）
        # 2. 确保梯度同步的一致性（所有 GPU 执行相同的 backward）
        # 3. 避免 DDP "marked as ready twice" 错误（只有一次 backward）
        # 4. 数学上等价：loss_sup.backward() + loss_unsup.backward() = (loss_sup + loss_unsup).backward()
        total_loss = loss_sup + loss_unsup_weighted
        
        # 🔥 统一反向传播（不需要 no_sync()，因为只有一次 backward）
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # 🔥 total_loss 会在作用域结束时自动释放，无需显式 del
        # del total_loss  # 已优化：减少不必要的释放
        
        # 计算总损失（仅用于日志记录）
        # 🔥 注意：loss_sup 和 loss_unsup_weighted 已经 backward 了，这里只用于日志
        loss_sum = loss_sup.detach() + loss_unsup_weighted.detach()
        
        # 🔥 优化：统计信息更新频率与日志同步（每 100 iter）
        # 这些统计信息主要用于日志，不需要每个 iteration 更新
        # 注意：num_pseudo_boxes 和 has_pseudo 需要保持更新以确保准确性
        
        # ========== 5. 日志记录（降低频率避免 GPU/CPU 同步） ==========
        # 🔥 关键修复：只在每 50 个 iteration 才进行 GPU/CPU 同步
        current_lr = optimizer.param_groups[0]["lr"]
        
        if i % 50 == 0:
            # 只在需要时才同步（减少 GPU/CPU 同步次数）
            loss_sum_val = float(loss_sum.detach())
            loss_sup_val = float(loss_sup.detach())
            loss_unsup_val = float(loss_unsup.detach()) if has_valid_pseudo else 0.0
                
            metric_logger.update(loss=loss_sum_val)
            metric_logger.update(loss_sup=loss_sup_val)
            metric_logger.update(loss_unsup=loss_unsup_val)
            metric_logger.update(lr=current_lr)
        
        # 🔥 更新 tqdm 进度条显示（只在主进程）
        if dist_utils.is_main_process() and isinstance(data_loader_iter, tqdm):
            postfix_dict = {
                'loss': f'{loss_sum_val:.4f}',
                'sup': f'{loss_sup_val:.4f}',
                'unsup': f'{loss_unsup_val:.4f}',
                'lr': f'{current_lr:.2e}',
                'pseudo': num_pseudo_boxes,
            }
            # 🔥 如果启用了原型筛选，显示过滤数量
            if use_proto_filter and proto_filtered > 0:
                postfix_dict['proto_flt'] = proto_filtered
            data_loader_iter.set_postfix(postfix_dict)
        
        # ========== 6. 优化器步进 ==========
        # 🔥 简化版：直接使用 scaler.step()，让 PyTorch 内部处理 inf 检测
        if scaler is not None:
            scaler.unscale_(optimizer)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm)
            optimizer.step()
           
        # ========== 7. EMA 更新（Semi-DETR 风格，改为基于 Epoch）==========
        # 🔥 改为基于 epoch 的 EMA warmup
        # momentum = min(target_momentum, 1 - (1 + warm_up_epochs) / (epoch + 1 + warm_up_epochs))
        if ema is not None:
            # 基于 epoch 的 EMA 动量计算
            total_epochs_for_ema = epoch + (i / total_iters_in_epoch)  # 精确的 epoch 进度（带小数）
            ema_decay = min(
                ema_momentum,
                1 - (1 + ema_warmup_epochs) / (total_epochs_for_ema + 1 + ema_warmup_epochs)
            )
            ema_decay = float(max(0.0, ema_decay))
            ema.update(student_model, decay=ema_decay)
            metric_logger.update(ema_momentum=ema_decay)
        
        # ========== 8. 学习率调度 ==========
        if self_lr_scheduler:
            optimizer = lr_scheduler.step(current_iter, optimizer)  # 🔥 使用 current_iter（已定义）
        elif lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()
        
        # 🔥 loss 变量会在作用域结束时自动释放，无需显式 del
        # del loss_sum, loss_sup, loss_unsup_weighted, loss_unsup  # 已优化：减少不必要的释放
    
    # 🔥 关闭 tqdm 进度条
    if dist_utils.is_main_process() and isinstance(data_loader_iter, tqdm):
        data_loader_iter.close()
    
    # 🔥 计算 epoch 耗时
    epoch_time = time.time() - epoch_start_time
    epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60)}s"
    
    # 🔥 构建返回统计信息
    result_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # 🔥 简化日志：显示关键损失和语义原型筛选统计
    if dist_utils.is_main_process():
        loss_total = result_stats.get('loss', 0.0)
        loss_sup = result_stats.get('loss_sup', 0.0)
        loss_unsup = result_stats.get('loss_unsup', 0.0)
        lr = result_stats.get('lr', 0.0)
        print(f"📊 Epoch {epoch} | 耗时: {epoch_time_str} | loss: {loss_total:.4f} (sup: {loss_sup:.4f}, unsup: {loss_unsup:.4f}) | lr: {lr:.2e}")
    
    return result_stats


def _load_and_preprocess_image(path: str, size: int = 800) -> Optional[torch.Tensor]:
    """从路径加载图像并预处理为 [3, H, W]，与训练时一致（Resize + Normalize）。"""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


def save_teacher_box_scores(
    teacher_model: nn.Module,
    image_list_path: str,
    device: torch.device,
    epoch: int,
    output_dir: Path,
    batch_size: int = 8,
) -> None:
    """
    每 N 个 epoch 用教师模型对指定无标注影像推理，保存 300 个目标框得分（不做绘图）。

    得分来自 pred_logits.sigmoid().max(dim=-1)[0]（每 query 的最大类别分数）。
    保存格式：{"epoch": N, "num_queries_per_image": 300, "images": [{"path": "...", "scores": [...]}, ...]}

    Args:
        teacher_model: 教师模型（EMA）
        image_list_path: 影像路径列表文件，每行一张（如 /data/DEIMv2/selected_images_30.txt）
        device: 设备
        epoch: 当前 epoch
        output_dir: 输出目录，保存到 output_dir/score_records/epoch_XXX.json
        batch_size: 推理时的 batch 大小
    """
    if not Path(image_list_path).exists():
        print(f"  ⚠️  影像列表文件不存在，跳过保存得分: {image_list_path}")
        return

    paths = []
    with open(image_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line if line.startswith('/') else str(Path(image_list_path).parent / line)
            if Path(p).exists():
                paths.append(p)
            else:
                print(f"  ⚠️  影像不存在，已跳过: {p}")

    if len(paths) == 0:
        print(f"  ⚠️  未找到有效影像，跳过保存得分")
        return

    rgb_list = []
    for p in paths:
        t = _load_and_preprocess_image(p)
        if t is not None:
            rgb_list.append(t)
        else:
            print(f"  ⚠️  无法加载影像，已跳过: {p}")
            rgb_list.append(None)

    # 过滤 None
    valid_indices = [i for i, t in enumerate(rgb_list) if t is not None]
    if len(valid_indices) == 0:
        return

    teacher_model.eval()
    all_scores_by_path = {paths[i]: None for i in valid_indices}

    with torch.no_grad():
        for start in range(0, len(valid_indices), batch_size):
            end = min(start + batch_size, len(valid_indices))
            batch_indices = valid_indices[start:end]
            batch_tensors = [rgb_list[i].to(device) for i in batch_indices]
            rgb_batch = torch.cat(batch_tensors, dim=0)
            out = teacher_model(rgb_batch, is_teacher=True)
            pred_logits = out.get('pred_logits')
            if pred_logits is None:
                continue
            scores = pred_logits.sigmoid().max(dim=-1)[0]  # [B, 300]
            scores_np = scores.cpu().numpy()
            for k, idx in enumerate(batch_indices):
                all_scores_by_path[paths[idx]] = scores_np[k].tolist()

    out_dir = Path(output_dir) / 'score_records'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f'epoch_{epoch:03d}.json'

    data = {
        'epoch': epoch,
        'num_queries_per_image': 300,
        'images': [
            {'path': p, 'scores': all_scores_by_path[p]}
            for p in paths
            if all_scores_by_path.get(p) is not None
        ],
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  📊 教师框得分已保存: {save_path} ({len(data['images'])} 张影像)")
