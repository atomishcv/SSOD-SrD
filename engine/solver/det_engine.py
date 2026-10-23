"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
import copy
from typing import Iterable

import torch
import torch.amp
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _to_device_obj(obj, device):
    """Safely move tensors/tv_tensors recursively to device; keep strings/ints as-is."""
    if isinstance(obj, dict):
        return {k: _to_device_obj(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_device_obj(v, device) for v in obj]
    # BoundingBoxes/Mask/Image also implement .to
    if hasattr(obj, 'to'):
        try:
            return obj.to(device)
        except Exception:
            return obj
    return obj

def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, use_hsv=False, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):       
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
        
        # 处理字典格式的samples（包含rgb和hsv）
        hsv_samples = None
        if isinstance(samples, dict) and 'rgb' in samples:
            rgb_samples = samples['rgb'].to(device)
            hsv_samples = samples['hsv'].to(device)
            samples = rgb_samples  # RGB作为主输入
        else:
            samples = samples.to(device)
        
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                # 将HSV直接传入模型
                outputs = model(samples, targets=targets, hsv=hsv_samples)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)  

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            # 将HSV直接传入模型
            outputs = model(samples, targets=targets, hsv=hsv_samples)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, use_hsv=False):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # 处理samples：可能是tensor或字典{'rgb': tensor, 'hsv': tensor}
        hsv_samples = None
        if isinstance(samples, dict):
            # 字典格式：包含rgb和hsv
            rgb_samples = samples['rgb'].to(device)
            hsv_samples = samples['hsv'].to(device)
            samples = rgb_samples  # RGB作为主输入
        else:
            samples = samples.to(device)
        
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in t.items()} for t in targets]

        # 将HSV传入模型
        outputs = model(samples, hsv=hsv_samples)

        # 🔥 检查模型输出是否包含 NaN/Inf
        if 'pred_boxes' in outputs:
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(f"⚠️  Warning: NaN/Inf detected in pred_boxes during evaluation!")
                print(f"   NaN count: {torch.isnan(outputs['pred_boxes']).sum().item()}")
                print(f"   Inf count: {torch.isinf(outputs['pred_boxes']).sum().item()}")
                # 清理异常值
                outputs['pred_boxes'] = torch.nan_to_num(outputs['pred_boxes'], nan=0.5, posinf=1.0, neginf=0.0)
        
        if 'pred_logits' in outputs:
            if torch.isnan(outputs['pred_logits']).any() or torch.isinf(outputs['pred_logits']).any():
                print(f"⚠️  Warning: NaN/Inf detected in pred_logits during evaluation!")
                print(f"   NaN count: {torch.isnan(outputs['pred_logits']).sum().item()}")
                print(f"   Inf count: {torch.isinf(outputs['pred_logits']).sum().item()}")
                # 清理异常值
                outputs['pred_logits'] = torch.nan_to_num(outputs['pred_logits'], nan=0.0, posinf=1e4, neginf=-1e4)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        # 🔥 过滤嵌套框（去除大框包小框的情况，如 small 在 big 框里）
        # 使用跨类别 NMS 和包含度检测，确保 small 和 big 重叠度较低，且不存在大框包小框的情况
        eval_nms_iou = 0.5  # 评估时的 NMS IoU 阈值
        eval_containment = 0.8  # 包含度阈值（小框被大框包含 80% 以上则抑制）
        results = filter_nested_boxes_in_results(results, iou_threshold=eval_nms_iou, containment_threshold=eval_containment)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator


def filter_nested_boxes_in_results(results, iou_threshold=0.5, containment_threshold=0.8):
    """
    过滤评估结果中的嵌套框（大框包小框的情况）
    
    场景：small 和 big 是同一物种，big 框内可能嵌套 small 框
    此函数会去除嵌套框，保留置信度更高的框
    
    Args:
        results: List of dicts, 每个 dict 包含 'boxes', 'scores', 'labels'
                boxes 格式为 xyxy（绝对坐标）
        iou_threshold: NMS 的 IoU 阈值，默认 0.5
        containment_threshold: 包含度阈值，默认 0.8（小框被大框包含 80% 以上则抑制）
        
    Returns:
        处理后的 results，已过滤嵌套框
    """
    from torchvision.ops import nms
    
    processed_results = []
    for result in results:
        boxes = result['boxes']  # [N, 4] xyxy 格式
        scores = result['scores']  # [N]
        labels = result['labels']  # [N]
        
        if len(boxes) == 0:
            processed_results.append(result)
            continue
        
        if len(boxes) == 1:
            processed_results.append(result)
            continue
        
        device = boxes.device
        N = boxes.shape[0]
        
        # ================================================================
        # Step 1: 标准 IoU NMS（处理重叠框）
        # ================================================================
        keep_indices = nms(boxes, scores, iou_threshold)
        keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
        keep_mask[keep_indices] = True
        
        # ================================================================
        # Step 2: 包含度检测（处理嵌套框）- 优化版本
        # ================================================================
        # 计算每个框的面积
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)  # [N]
        
        # 对保留的框进行嵌套检测
        kept_indices = keep_indices.tolist()
        if len(kept_indices) <= 1:
            # 如果只有0或1个框，无需检测嵌套
            suppressed = set()
        else:
            suppressed = set()
            
            # 🔥 优化：使用向量化计算，减少循环开销
            # 只对保留的框进行检测
            kept_boxes = boxes[keep_indices]  # [K, 4]
            kept_areas = areas[keep_indices]  # [K]
            K = len(kept_indices)
            
            # 向量化计算所有框对之间的交集（只计算上三角，避免重复）
            kept_x1, kept_y1 = kept_boxes[:, 0], kept_boxes[:, 1]
            kept_x2, kept_y2 = kept_boxes[:, 2], kept_boxes[:, 3]
            
            # 广播计算交集 [K, K]
            xx1 = torch.max(kept_x1[:, None], kept_x1[None, :])
            yy1 = torch.max(kept_y1[:, None], kept_y1[None, :])
            xx2 = torch.min(kept_x2[:, None], kept_x2[None, :])
            yy2 = torch.min(kept_y2[:, None], kept_y2[None, :])
            
            inter_w = torch.clamp(xx2 - xx1, min=0)
            inter_h = torch.clamp(yy2 - yy1, min=0)
            inter_area = inter_w * inter_h  # [K, K]
            
            # 计算包含度矩阵：交集 / 较小框的面积
            smaller_area = torch.min(kept_areas[:, None], kept_areas[None, :])
            containment_matrix = inter_area / (smaller_area + 1e-8)  # [K, K]
            
            # 只考虑上三角矩阵（i < j），避免重复计算和自比较
            # 对于每个框对 (i, j)，如果包含度 >= 阈值，抑制置信度较低的框 j
            for i in range(K):
                for j in range(i + 1, K):
                    if containment_matrix[i, j] >= containment_threshold:
                        # 抑制置信度较低的框（j 对应原始索引 kept_indices[j]）
                        suppressed.add(kept_indices[j])
        
        # 更新 keep_mask
        for idx in suppressed:
            keep_mask[idx] = False
        
        # 应用过滤
        processed_results.append({
            'boxes': boxes[keep_mask],
            'scores': scores[keep_mask],
            'labels': labels[keep_mask],
        })
    
    return processed_results


def apply_nms_to_results(results, iou_threshold=0.5):
    """
    对预测结果应用 NMS，将所有类别合并为单一类别后进行 NMS。
    
    Args:
        results: List of dicts, 每个 dict 包含 'boxes', 'scores', 'labels'
        iou_threshold: NMS 的 IoU 阈值
        
    Returns:
        处理后的 results，所有类别合并为 0
    """
    processed_results = []
    for result in results:
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']
        
        if len(boxes) == 0:
            processed_results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': torch.zeros_like(labels)  # 合并为类别 0
            })
            continue
        
        # 所有类别合并，执行 NMS
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        
        processed_results.append({
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'labels': torch.zeros(len(keep_indices), dtype=labels.dtype, device=labels.device)  # 全部归为类别 0
        })
    
    return processed_results


# 🔥 全局缓存：避免每个 epoch 重复创建合并后的 COCO GT
_merged_coco_gt_cache = {}


def create_merged_class_coco_gt(coco_gt_original, use_cache=True):
    """
    创建一个将所有类别合并为单一类别的 COCO GT 对象。
    
    支持以下场景：
    1. 真值有多个类别 → 合并为单一类别（category_id=0）
    2. 真值只有一个类别 → 直接使用，确保 category_id=0
    
    Args:
        coco_gt_original: 原始 COCO GT 对象
        use_cache: 是否使用缓存（默认 True，避免重复创建）
        
    Returns:
        新的 COCO GT 对象，只有一个类别（category_id=0）
    """
    global _merged_coco_gt_cache
    
    # 🔥 使用缓存：通过 id 标识原始 coco_gt 对象
    cache_key = id(coco_gt_original)
    if use_cache and cache_key in _merged_coco_gt_cache:
        return _merged_coco_gt_cache[cache_key]
    
    from faster_coco_eval import COCO
    
    # 深拷贝原始数据集（仅在首次创建时执行）
    merged_dataset = copy.deepcopy(coco_gt_original.dataset)
    
    # 获取原始类别信息
    original_categories = merged_dataset.get('categories', [])
    num_original_cats = len(original_categories)
    
    # 🔥 打印原始类别信息（便于调试）
    if dist_utils.is_main_process():
        cat_names = [c.get('name', f"id={c.get('id')}") for c in original_categories]
        print(f"📋 真值类别信息: {num_original_cats} 个类别 - {cat_names}")
    
    # 1. 修改 categories：只保留一个类别（统一为 id=0）
    merged_dataset['categories'] = [{'id': 0, 'name': 'object', 'supercategory': 'object'}]
    
    # 2. 修改所有 annotations 的 category_id 为 0（确保与预测类别对齐）
    for ann in merged_dataset['annotations']:
        ann['category_id'] = 0
    
    # 3. 创建新的 COCO 对象
    merged_coco_gt = COCO()
    merged_coco_gt.dataset = merged_dataset
    merged_coco_gt.createIndex()
    
    if dist_utils.is_main_process():
        print(f"✅ 真值已转换为单类别格式 (category_id=0)，共 {len(merged_dataset['annotations'])} 个标注")
    
    # 🔥 缓存结果
    if use_cache:
        _merged_coco_gt_cache[cache_key] = merged_coco_gt
    
    return merged_coco_gt


# 🔥 全局缓存：缓存 CocoEvaluator 模板，避免每次重建
_coco_evaluator_cache = {}


def _create_evaluator_for_merged_gt(merged_coco_gt, iou_types):
    """
    为合并后的 COCO GT 创建评估器（使用缓存避免重复深拷贝）。
    
    这个函数创建一个轻量级的评估器，复用已缓存的 coco_gt 对象。
    """
    global _coco_evaluator_cache
    
    cache_key = id(merged_coco_gt)
    
    # 🔥 检查是否有缓存的评估器模板
    if cache_key in _coco_evaluator_cache:
        # 复用缓存的评估器，只需要 cleanup 重置状态
        evaluator = _coco_evaluator_cache[cache_key]
        evaluator.cleanup()
        return evaluator
    
    # 首次创建：正常实例化（会进行一次深拷贝）
    evaluator = CocoEvaluator(merged_coco_gt, iou_types)
    
    # 缓存评估器
    _coco_evaluator_cache[cache_key] = evaluator
    
    return evaluator


@torch.no_grad()
def evaluate_merged_classes(
    model: torch.nn.Module, 
    criterion: torch.nn.Module, 
    postprocessor, 
    data_loader, 
    coco_gt,  # 原始 COCO GT 对象（从 evaluator.coco_gt 获取）
    device, 
    use_hsv=False,
    conf_thresh=0.3,  # 🔥 置信度阈值，低于此值的预测框将被过滤
    nms_iou_threshold=0.6,  # 🔥 提高默认值：0.5 可能过低，会过滤掉太多重叠框，导致 precision 降低
    containment_threshold=0.8  # 包含度阈值（小框被大框包含的比例）
):
    """
    将模型预测的多个类别合并为单一类别，与单类别真值进行评估。
    
    核心处理流程：
    1. 模型预测可能输出多个类别（如 0=small, 1=big）
    2. 将所有预测类别合并为单一类别 (category_id=0)
    3. 使用 NMS + 嵌套框过滤去除重复检测框
    4. 与单类别真值 (category_id=0) 计算 COCO mAP
    
    适用场景：
    - 模型输出多个类别（如 small/big），但真值只有一个类别
    - 多个类别实际上是同一物种（如 small/big 只是尺寸不同）
    - 希望评估定位能力而非分类能力
    - 存在大框套小框的嵌套检测需要用 NMS 去重
    
    Args:
        model: 待评估的模型
        criterion: 损失函数（用于保持接口一致）
        postprocessor: 后处理器
        data_loader: 验证数据加载器
        coco_gt: 原始 COCO GT 对象（可能是单类别或多类别）
        device: 设备
        use_hsv: 是否使用 HSV 输入
        conf_thresh: 置信度阈值（默认 0.3）
            - 低于此值的预测框将被过滤，不参与评估
            - 0.0: 不过滤，使用所有预测框
            - 0.3-0.5: 推荐值，过滤低质量预测
            - 0.5+: 较严格，只保留高置信度预测
        nms_iou_threshold: NMS IoU 阈值（默认 0.6）
            - 0.5: 较严格，会过滤更多重叠框，可能导致 precision 降低
            - 0.6-0.7: 推荐值，平衡 precision 和 recall
            - 0.8+: 较宽松，可能保留过多重复检测
        containment_threshold: 包含度阈值（默认 0.8）
            - 用于过滤嵌套框：小框被大框包含超过此比例时，抑制置信度较低的框
        
    Returns:
        stats: 评估统计信息
        coco_evaluator: COCO 评估器对象
    """
    model.eval()
    criterion.eval()
    
    if dist_utils.is_main_process():
        print(f"\n{'='*60}")
        print(f"🔄 合并类别评估模式")
        print(f"   - 模型预测类别: 合并为单一类别 (category_id=0)")
        print(f"   - 置信度阈值: {conf_thresh}")
        print(f"   - NMS IoU 阈值: {nms_iou_threshold}")
        print(f"   - 嵌套框过滤阈值: {containment_threshold}")
        print(f"{'='*60}")
    
    # 🔥 创建合并类别后的 COCO GT（使用缓存，首次之后直接返回缓存结果）
    # 无论真值是单类别还是多类别，都统一转换为 category_id=0
    merged_coco_gt = create_merged_class_coco_gt(coco_gt, use_cache=True)
    
    # 🔥 创建评估器（使用缓存，避免每次都深拷贝 + 重建索引）
    iou_types = ['bbox']
    coco_evaluator = _create_evaluator_for_merged_gt(merged_coco_gt, iou_types)
    
    # 🔥 关键：修改 label2category 映射，将模型预测的所有类别都映射到 category_id=0
    # 这样模型预测的 small(0) 和 big(1) 都会映射到真值的单一类别
    coco_evaluator.label2category = {i: 0 for i in range(100)}  # 支持最多 100 个类别
    
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test (Merged Classes):'
    
    # 🔥 统计变量（用于调试）
    total_pred_before_conf_thresh = 0  # 置信度过滤前的数量
    total_pred_before_merge = 0
    total_pred_after_cross_class_nms = 0  # 跨类别 NMS 后的数量
    total_pred_after_nms = 0
    class_0_count = 0
    class_1_count = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # 处理 samples
        hsv_samples = None
        if isinstance(samples, dict):
            rgb_samples = samples['rgb'].to(device)
            hsv_samples = samples['hsv'].to(device)
            samples = rgb_samples
        else:
            samples = samples.to(device)
        
        targets = [{k: v.to(device) if hasattr(v, 'to') else v for k, v in t.items()} for t in targets]

        # 前向推理
        outputs = model(samples, hsv=hsv_samples)

        # 检查 NaN/Inf
        if 'pred_boxes' in outputs:
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(f"⚠️  Warning: NaN/Inf detected in pred_boxes during evaluation!")
                outputs['pred_boxes'] = torch.nan_to_num(outputs['pred_boxes'], nan=0.5, posinf=1.0, neginf=0.0)
        
        if 'pred_logits' in outputs:
            if torch.isnan(outputs['pred_logits']).any() or torch.isinf(outputs['pred_logits']).any():
                print(f"⚠️  Warning: NaN/Inf detected in pred_logits during evaluation!")
                outputs['pred_logits'] = torch.nan_to_num(outputs['pred_logits'], nan=0.0, posinf=1e4, neginf=-1e4)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        # 后处理得到预测结果
        results = postprocessor(outputs, orig_target_sizes)
        
        # 🔥 统计置信度过滤前的数量
        for result in results:
            total_pred_before_conf_thresh += len(result['labels'])
        
        # 🔥 置信度阈值过滤：过滤掉低置信度的预测框
        if conf_thresh > 0:
            filtered_results = []
            for result in results:
                if len(result['scores']) > 0:
                    keep_mask = result['scores'] >= conf_thresh
                    filtered_result = {
                        'boxes': result['boxes'][keep_mask],
                        'scores': result['scores'][keep_mask],
                        'labels': result['labels'][keep_mask],
                    }
                    filtered_results.append(filtered_result)
                else:
                    filtered_results.append(result)
            results = filtered_results
        
        # 🔥 统计合并前的类别分布
        for result in results:
            if len(result['labels']) > 0:
                labels = result['labels']
                total_pred_before_merge += len(labels)
                class_0_count += (labels == 0).sum().item()
                class_1_count += (labels == 1).sum().item()
        
        # 🔥 核心步骤0：先进行跨类别 NMS（在合并类别之前，去除 small 和 big 之间的重叠框）
        # 这一步很重要：可以提前去除 small 和 big 类别之间的重叠框，避免大框套小框的问题
        from torchvision.ops import nms
        processed_results = []
        for result in results:
            boxes = result['boxes']  # [N, 4] xyxy 格式
            scores = result['scores']  # [N]
            labels = result['labels']  # [N]
            
            if len(boxes) == 0:
                processed_results.append(result)
                continue
            
            # 🔥 跨类别 NMS：不区分类别，所有重叠框都会被去除
            keep_indices = nms(boxes, scores, nms_iou_threshold)
            
            processed_results.append({
                'boxes': boxes[keep_indices],
                'scores': scores[keep_indices],
                'labels': labels[keep_indices],
            })
        
        results = processed_results
        
        # 🔥 统计跨类别 NMS 后的预测数量
        for result in results:
            total_pred_after_cross_class_nms += len(result['labels'])
        
        # 🔥 核心步骤1：合并所有预测类别为单一类别 (category_id=0)
        # 模型可能预测 small(0) 和 big(1)，都合并为 0
        for result in results:
            if len(result['labels']) > 0:
                result['labels'] = torch.zeros_like(result['labels'])  # 合并为类别 0
        
        # 🔥 核心步骤2：应用嵌套框过滤去除重复检测框（跨类别 NMS 之后，再处理嵌套情况）
        # filter_nested_boxes_in_results 内部包含：
        #   1. 标准 IoU NMS（处理重叠框）
        #   2. 包含度检测（处理大框套小框的嵌套情况）
        results = filter_nested_boxes_in_results(
            results, 
            iou_threshold=nms_iou_threshold, 
            containment_threshold=containment_threshold
        )
        
        # 🔥 统计 NMS 后的预测数量
        for result in results:
            total_pred_after_nms += len(result['labels'])

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # 同步统计
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # 🔥 打印类别合并统计信息
    if dist_utils.is_main_process():
        print(f"\n📊 类别合并统计:")
        print(f"   - PostProcessor输出总数: {total_pred_before_conf_thresh}")
        if conf_thresh > 0:
            print(f"   - 置信度过滤后预测数 (≥{conf_thresh}): {total_pred_before_merge}")
            print(f"   - 置信度过滤去除: {total_pred_before_conf_thresh - total_pred_before_merge} 个低置信度框")
        else:
            print(f"   - 置信度过滤后预测数 (无过滤): {total_pred_before_merge}")
        print(f"   - 其中 small(0): {class_0_count}, big(1): {class_1_count}")
        print(f"   - 跨类别NMS后预测数: {total_pred_after_cross_class_nms}")
        print(f"   - 跨类别NMS去除: {total_pred_before_merge - total_pred_after_cross_class_nms} 个重叠框")
        print(f"   - 嵌套框过滤后预测数: {total_pred_after_nms}")
        print(f"   - 嵌套框过滤去除: {total_pred_after_cross_class_nms - total_pred_after_nms} 个嵌套框")
        print(f"   - 总共去除重复框: {total_pred_before_merge - total_pred_after_nms}")
    
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # 汇总结果
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    return stats, coco_evaluator
