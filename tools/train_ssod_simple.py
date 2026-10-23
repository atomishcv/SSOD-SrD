"""
简洁实用的半监督目标检测训练框架 (SSOD)

核心特性:
- Mean-Teacher (EMA) 学生/教师模型
- 弱/强增强分离的无标签数据
- 基于迭代的无监督损失 warmup + EMA 动量 warmup
- 自适应阈值伪标签生成 + NMS + 尺寸过滤
- 可选的教师伪标签可视化
- 最少外部依赖（torch/torchvision）

使用示例:
    python tools/train_ssod_simple.py \
        --config configs/deimv2_ssod/deimv2_dinov3_x_coco.yml \
        --checkpoint outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth \
        --gpus 1,2 \
        --epochs 100
"""

import os
import sys
import argparse
import copy
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import nms as torchvision_nms

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.core import YAMLConfig
from engine.misc import dist_utils


# ========================================
# 工具函数
# ========================================

def setup_environment(gpu_ids: str):
    """设置训练环境"""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    os.environ['NCCL_TIMEOUT'] = '10800'
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'


def move_to_device(obj, device):
    """递归移动对象到设备"""
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [move_to_device(v, device) for v in obj]
    if hasattr(obj, 'to'):
        return obj.to(device)
    return obj

def _set_dynamic_pos_embed(model: nn.Module):
    """如模型支持，将位置编码设置为动态尺寸以匹配不同输入大小"""
    try:
        enc = getattr(model, "encoder", None)
        if enc is not None and hasattr(enc, "eval_spatial_size"):
            setattr(enc, "eval_spatial_size", None)
            print("🔧 设置动态位置编码: encoder.eval_spatial_size=None")
    except Exception as _:
        pass

@torch.no_grad()
def _sync_teacher_from_student(teacher: nn.Module, student: nn.Module):
    """确保teacher与student完全一致（含参数与缓冲），用于初始化或调试"""
    try:
        teacher.load_state_dict(student.state_dict(), strict=True)
        for p in teacher.parameters():
            p.requires_grad_(False)
    except Exception as e:
        print(f"[WARN] 同步teacher失败: {e}")

@torch.no_grad()
def _fingerprint(model: nn.Module) -> float:
    """计算模型参数与浮点缓冲的简易指纹（平均绝对值之和），用于调试对齐"""
    s = 0.0
    count = 0
    for p in model.parameters():
        if p.dtype.is_floating_point:
            val = float(p.detach().abs().mean())
            if not (torch.isfinite(torch.tensor(val)).item()):
                # 参数中包含 inf 或 nan
                continue
            s += val
            count += 1
    for b in model.buffers():
        if torch.is_floating_point(b):
            val = float(b.detach().abs().mean())
            if not (torch.isfinite(torch.tensor(val)).item()):
                # 缓冲中包含 inf 或 nan
                continue
            s += val
            count += 1
    return s

# ========================================
# 日志工具
# ========================================

class SmoothedValue:
    """指数平滑的数值记录器"""
    def __init__(self, window_size=20):
        self.deque = []
        self.window_size = window_size
        self.total = 0.0
        self.count = 0
    
    def update(self, value: float):
        self.deque.append(float(value))
        self.total += float(value)
        self.count += 1
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
    
    @property
    def avg(self):
        return sum(self.deque) / max(1, len(self.deque))
    
    @property
    def global_avg(self):
        return self.total / max(1, self.count)


class MetricLogger:
    """指标记录器"""
    def __init__(self):
        self.meters = {}
    
    def add_meter(self, name: str):
        self.meters[name] = SmoothedValue()
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.add_meter(k)
            self.meters[k].update(v)
    
    def format(self):
        parts = []
        for k, m in self.meters.items():
            parts.append(f"{k}: {m.avg:.4f}")
        return " | ".join(parts)


# ========================================
# 坐标转换工具
# ========================================

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """cxcywh -> xyxy"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def scale_xyxy_norm_to_pixels(xyxy: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """归一化 xyxy -> 像素 xyxy"""
    scale = torch.tensor([W, H, W, H], dtype=xyxy.dtype, device=xyxy.device)
    return xyxy * scale


# ========================================
# 调度器
# ========================================

@dataclass
class Schedules:
    """训练调度器"""
    warmup_iters: int = 12000         # λ_u 从 0->1 的迭代数
    ema_target: float = 0.999         # 目标 EMA 动量
    ema_warmup: int = 100             # EMA 动量从 0->target 的迭代数
    
    def lambda_u(self, global_iter: int, base_lambda: float) -> float:
        """计算当前无监督损失权重"""
        if self.warmup_iters <= 0:
            return base_lambda
        ramp = min(1.0, float(global_iter) / float(self.warmup_iters))
        return base_lambda * ramp
    
    def ema_momentum(self, global_iter: int) -> float:
        """计算当前 EMA 动量"""
        if self.ema_warmup <= 0:
            return self.ema_target
        ramp = min(1.0, float(global_iter) / float(self.ema_warmup))
        return self.ema_target * ramp


# ========================================
# EMA 模型
# ========================================

class ModelEMA(nn.Module):
    """指数移动平均模型 (教师模型)"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
    
    @torch.no_grad()
    def set_momentum(self, decay: float):
        """动态调整 EMA 动量"""
        self.decay = decay
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """使用学生模型更新教师模型
        策略：
        - 参数: 指数滑动平均 (EMA)
        - BN 等浮点缓冲区: 直接对齐复制（不做 EMA），避免统计滞后带来的额外偏移
        """
        d = self.decay
        # 1) 参数做 EMA
        ema_params = dict(self.module.named_parameters())
        model_params = dict(model.named_parameters())
        for name, p in model_params.items():
            if name in ema_params and p.dtype.is_floating_point:
                ema_p = ema_params[name]
                ema_p.mul_(d).add_(p, alpha=1.0 - d)
        # 2) 浮点缓冲（如 BN running_mean/var）直接拷贝
        ema_buffers = dict(self.module.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, buf in model_buffers.items():
            if name in ema_buffers and torch.is_floating_point(buf):
                ema_buffers[name].copy_(buf)


# ========================================
# 伪标签生成配置
# ========================================

@dataclass
class PseudoCfg:
    """伪标签生成配置"""
    score_thresh: float = 0.3        # 固定阈值
    input_size: int = 640            # 输入尺寸（用于可视化）
    vis_dir: Optional[str] = None    # 可视化保存目录


class PseudoLabeler:
    """伪标签生成器"""
    def __init__(self, cfg: PseudoCfg):
        self.cfg = cfg
        self._vis_counter = 0
    
    @torch.no_grad()
    def __call__(self, model: nn.Module, images: torch.Tensor, is_teacher: bool = True):
        """
        生成伪标签
        
        Args:
            model: 模型（教师或学生）
            images: 图像 [B, 3, H, W]
            is_teacher: 是否为教师模型
        
        Returns:
            伪标签列表 [{'boxes': [N,4], 'labels': [N], 'scores': [N]}]
            模型输出 {'pred_logits', 'pred_boxes'}
        """
        # 前向传播
        # 计算模型权重参数的平均值
        weight_params = [p for p in model.parameters() if p.data.is_floating_point()]
        if weight_params:
            avg_weight = torch.mean(torch.cat([p.data.flatten() for p in weight_params]))
            print(f"模型权重参数的平均值: {avg_weight.item():.6f}")
        outputs = model(images)
        logits = outputs['pred_logits']  # [B, Q, C]
        boxes = outputs['pred_boxes']     # [B, Q, 4] (norm cxcywh)
        
        B, Q, C = logits.shape
        results = []
        
        # 计算置信度和标签
        probs = logits.sigmoid()  # [B, Q, C]
        conf, labels = probs.max(dim=-1)  # [B, Q]
        
        for b in range(B):
            conf_b = conf[b]
            labels_b = labels[b]
            boxes_b = boxes[b]
            
            # ========== 简单阈值过滤（固定阈值 0.3）==========
            thr = 0.3
            keep = conf_b >= thr
            
            if keep.sum() == 0:
                results.append({
                    'boxes': boxes_b.new_zeros((0, 4)),
                    'labels': labels_b.new_zeros((0,), dtype=torch.long),
                    'scores': conf_b.new_zeros((0,)),
                })
                continue
            
            boxes_k = boxes_b[keep]
            labels_k = labels_b[keep]
            scores_k = conf_b[keep]
            
            results.append({
                'boxes': boxes_k.detach(),
                'labels': labels_k.detach(),
                'scores': scores_k.detach(),
            })
        
        # 可选：教师模型可视化
        if is_teacher and self.cfg.vis_dir:
            try:
                os.makedirs(self.cfg.vis_dir, exist_ok=True)
                self._dump_teacher_vis(images, results)
            except Exception as e:
                print(f"[警告] 教师可视化失败: {e}")
        
        return results, outputs
    
    def _dump_teacher_vis(self, images: torch.Tensor, pseudo):
        """保存教师伪标签可视化"""
        import cv2
        import numpy as np
        
        B = images.shape[0]
        for b in range(min(B, 4)):  # 最多保存2张
            # 反归一化图像
            img = images[b].detach().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img * std + mean).clamp(0, 1)
            img = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            H, W = img.shape[:2]
            vis = img.copy()
            
            # 绘制伪标签
            p = pseudo[b]
            boxes = p['boxes']
            scores = p['scores']
            
            if len(boxes) > 0:
                xyxy = cxcywh_to_xyxy(boxes)
                xyxy_pix = scale_xyxy_norm_to_pixels(xyxy, W, H).round().int().cpu().numpy()
                
                for (x1, y1, x2, y2), s in zip(xyxy_pix, scores.cpu().numpy()):
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{s:.2f}", (x1, max(0, y1-5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 保存
            out_path = os.path.join(self.cfg.vis_dir, f"teacher_{self._vis_counter:06d}.jpg")
            cv2.imwrite(out_path, vis)
            self._vis_counter += 1


# ========================================
# 损失计算
# ========================================

def compute_supervised_loss(student, criterion, images, targets, device, scaler=None, vis_dir=None, vis_threshold=0.3, vis_counter=[0]):
    """计算监督损失，并可视化学生模型输出"""
    import os
    import cv2
    import numpy as np

    def _dump_student_vis(images, outputs, vis_dir, vis_counter, threshold=0.3):
        os.makedirs(vis_dir, exist_ok=True)

        B = images.shape[0]
        for b in range(min(B, 4)):
            img = images[b].detach().cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img * std + mean).clamp(0, 1)
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            H, W = img_np.shape[:2]
            vis = img_np.copy()
            # 假设 outputs 的格式为 dict，并含有 'boxes', 'scores'
            single_det = outputs[b]
            boxes = single_det['boxes']
            scores = single_det['scores']

            # 只保留分数大于阈值的
            if len(boxes) > 0:
                mask = scores > threshold
                boxes = boxes[mask]
                scores = scores[mask]
                if len(boxes) > 0:
                    # 转为像素坐标
                    xyxy = cxcywh_to_xyxy(boxes)
                    xyxy_pix = scale_xyxy_norm_to_pixels(xyxy, W, H).round().int().cpu().numpy()
                    for (x1, y1, x2, y2), s in zip(xyxy_pix, scores.cpu().numpy()):
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色学生
                        cv2.putText(vis, f"{s:.2f}", (x1, max(0, y1-5)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            out_path = os.path.join(vis_dir, f"student_{vis_counter[0]:06d}.jpg")
            cv2.imwrite(out_path, vis)
            vis_counter[0] += 1

    images = images.to(device)
    targets = move_to_device(targets, device)
    
    if scaler is not None:
        with torch.autocast(device_type=str(device).split(':')[0], cache_enabled=True):
            outputs = student(images, targets=targets)
        with torch.autocast(device_type=str(device).split(':')[0], enabled=False):
            loss_dict = criterion(outputs, targets)
    else:
        outputs = student(images, targets=targets)
        loss_dict = criterion(outputs, targets)

    # 可视化部分
    # 输出可视化目录由 vis_dir 参数决定(外部传入/控制)
    if vis_dir is not None:
        try:
            _dump_student_vis(images, outputs, vis_dir, vis_counter, threshold=vis_threshold)
        except Exception as e:
            print(f"[警告] 学生可视化失败: {e}")

    loss = sum(loss_dict.values())
    return loss, loss_dict


def compute_unsupervised_loss(student, criterion, images, pseudo_labels, device, scaler=None):
    """计算无监督损失"""
    images = images.to(device)
    
    # 转换伪标签为 targets 格式
    targets_pseudo = []
    for pl in pseudo_labels:
        if len(pl['labels']) == 0:
            targets_pseudo.append({
                'boxes': torch.empty((0, 4), device=device),
                'labels': torch.empty((0,), dtype=torch.long, device=device),
            })
        else:
            targets_pseudo.append({
                'boxes': pl['boxes'].to(device),
                'labels': pl['labels'].to(device),
            })
    
    # 前向传播
    if scaler is not None:
        with torch.autocast(device_type=str(device).split(':')[0], cache_enabled=True):
            outputs = student(images, targets=targets_pseudo)
        with torch.autocast(device_type=str(device).split(':')[0], enabled=False):
            loss_dict = criterion(outputs, targets_pseudo)
    else:
        outputs = student(images, targets=targets_pseudo)
        loss_dict = criterion(outputs, targets_pseudo)
    
    loss = sum(loss_dict.values())
    
    # 添加 unsup 后缀
    loss_dict = {f"{k}_unsup": v for k, v in loss_dict.items()}
    
    return loss, loss_dict


# ========================================
# 训练一个 epoch
# ========================================

def train_one_epoch(
    student, teacher, criterion, optimizer,
    group_reader,
    device, epoch, cfg,
    pseudo_labeler: PseudoLabeler,
    schedules: Schedules,
    base_lambda_u: float = 2.0,
    scaler=None,
    debug_compare_first_iter: bool = False,
):
    """训练一个 epoch"""
    student.train()
    teacher.module.eval()
    
    print(f"\n{'='*80}")
    print(f"🚀 Epoch {epoch} 开始训练")
    print(f"{'='*80}\n")
    
    logger = MetricLogger()
    max_iters = len(group_reader)
    epoch_start_iter = epoch * max_iters
    
    start_time = time.time()
    
    # 训练循环
    for i, groups in enumerate(group_reader):
        global_iter = epoch_start_iter + i
        
        # ========== 1. 获取数据 ==========
        images_l, targets_l = groups['sup']
        images_u_w, targets_u_w = groups['unsup_teacher']  # 弱增强
        images_u_s, targets_u_s = groups['unsup_student']  # 强增强
        
        # ---------- 可选：首轮等价性调试（同输入+同分支） ----------
        # 注意：必须在任何前向传播之前进行对比，否则 BN 层的统计量会被更新
        if debug_compare_first_iter and i == 0:
            with torch.no_grad():
                # 指纹：检查师生初始是否完全一致（参数+BN缓冲）
                try:
                    fp_s = _fingerprint(student)
                    fp_t = _fingerprint(teacher.module)
                    print(f"🔍 [Debug] Fingerprint student={fp_s:.6f}, teacher={fp_t:.6f}, Δ={abs(fp_s - fp_t):.6f}")
                except Exception as e:
                    print(f"[WARN] 指纹计算失败: {e}")
                # 选择同一输入（使用强增强的 images_u_s 以匹配真实尺寸）
                same_inp = images_u_s.to(device)
                # 暂存训练状态
                stu_was_train = student.training
                student.eval()
                # 强制两者都使用相同的 forward 路径（不使用 teacher 特殊逻辑）
                try:
                    out_s = student(same_inp, is_teacher=False)
                except TypeError:
                    out_s = student(same_inp)
                tm = teacher.module
                try:
                    out_t = tm(same_inp, is_teacher=False)
                except TypeError:
                    out_t = tm(same_inp)
                # 计算差异
                try:
                    diff_logits = (out_s['pred_logits'] - out_t['pred_logits']).abs().mean().item()
                    diff_boxes  = (out_s['pred_boxes']  - out_t['pred_boxes']).abs().mean().item()
                    print(f"🔍 [Debug] First iter S/T Δ logits={diff_logits:.6f}, Δ boxes={diff_boxes:.6f}")
                except Exception as e:
                    print(f"[WARN] 无法计算S/T差异: {e}")
                # 恢复训练状态
                if stu_was_train:
                    student.train()
        
        # ========== 2. 计算当前调度参数 ==========
        lambda_u = schedules.lambda_u(global_iter, base_lambda_u)
        ema_momentum = schedules.ema_momentum(global_iter)
        teacher.set_momentum(ema_momentum)
        
        # ========== 3. 监督损失 ==========
        loss_sup, loss_dict_sup = compute_supervised_loss(
            student, criterion, images_l, targets_l, device, scaler,
            vis_dir=cfg.get("vis_dir", None) if isinstance(cfg, dict) else getattr(cfg, "vis_dir", None)
        )
        
        # ========== 4. 无监督损失 ==========
        # 4.1 教师生成伪标签（弱增强）
        with torch.no_grad():
            pseudo_labels, _ = pseudo_labeler(
                teacher.module, images_u_w.to(device), is_teacher=True
            )
        
        # 4.2 学生在强增强上计算损失
        loss_unsup, loss_dict_unsup = compute_unsupervised_loss(
            student, criterion, images_u_s, pseudo_labels, device, scaler
        )
        
        # ========== 5. 总损失 ==========
        loss = loss_sup + lambda_u * loss_unsup
        
        # ========== 6. 反向传播 ==========
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # ========== 7. 更新教师模型 (EMA) ==========
        teacher.update(student)
        
        # ========== 8. 记录 ==========
        logger.update(
            loss_sup=float(loss_sup.detach()),
            loss_unsup=float(loss_unsup.detach()),
            lr=optimizer.param_groups[0]['lr'],
            lambda_u=lambda_u,
            ema_mom=ema_momentum
        )
        
        # 打印进度
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (max_iters - i - 1)
            print(f"[{i+1}/{max_iters}] {logger.format()} | "
                  f"ETA: {eta/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✅ Epoch {epoch} 完成 (用时 {total_time/60:.1f} 分钟)")
    print(f"   {logger.format()}")
    print(f"{'='*80}\n")
    
    return {k: m.global_avg for k, m in logger.meters.items()}


# ========================================
# 评估
# ========================================

@torch.no_grad()
def evaluate(model, val_loader, postprocessor, evaluator, device):
    """评估模型"""
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"📊 开始评估")
    print(f"{'='*80}\n")
    
    for images, targets in val_loader:
        images = images.to(device)
        targets = move_to_device(targets, device)
        
        outputs = model(images)
        
        # 后处理
        orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)
        
        # 收集结果
        evaluator.update(results, targets)
    
    # 计算指标
    stats = evaluator.synchronize_between_processes()
    stats = evaluator.summarize()
    
    print(f"\n{'='*80}")
    print(f"✅ 评估完成")
    if stats:
        print(f"   AP: {stats[0]:.4f}")
    print(f"{'='*80}\n")
    
    return stats


# ========================================
# 主训练函数
# ========================================

def main(local_rank, args):
    """主训练函数"""
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    # 设置分布式
    if args.num_gpus > 1:
        os.environ['RANK'] = str(local_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(args.num_gpus)
        dist_utils.setup_distributed(0, 'builtin', seed=args.seed)
    
    # 加载配置
    print(f"\n{'='*80}")
    print(f"⚙️  加载配置")
    print(f"{'='*80}")
    print(f"   配置文件: {args.config}")
    print(f"   检查点: {args.checkpoint}")
    print(f"   输出目录: {args.output_dir}")
    print(f"   设备: {device}")
    print(f"{'='*80}\n")
    
    cfg = YAMLConfig(args.config)
    
    # 加载模型和检查点
    print(f"📦 加载模型...")
    model = cfg.model.to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"加载检查点权重...")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        
        # 处理不同的检查点格式
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'ema' in ckpt:
            # 如果有 EMA 权重，优先使用
            ema_state = ckpt['ema']
            if isinstance(ema_state, dict) and 'module' in ema_state:
                state_dict = ema_state['module']
            else:
                state_dict = ema_state
        else:
            state_dict = ckpt
        
        # 移除 'module.' 前缀（如果有）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'eval_spatial_size'):
            model.encoder.eval_spatial_size = None
            print("✅ 已设置动态位置编码（eval_spatial_size=None）")
        
        print(f"✅ 加载检查点: {args.checkpoint}")
        if 'epoch' in ckpt:
            print(f"   - Epoch: {ckpt['epoch']}")
        if 'ap' in ckpt:
            print(f"   - AP: {ckpt['ap']:.4f}")
    
    # 创建 EMA 教师模型
    schedules = Schedules(
        warmup_iters=args.warmup_iters,
        ema_target=args.ema_momentum,
        ema_warmup=args.ema_warmup
    )
    teacher = ModelEMA(model, decay=schedules.ema_target)
    print(f"✅ 创建 EMA 教师模型 (动量: {schedules.ema_target})")
    # 初始时强制 teacher 完全对齐 student，包含BN等缓冲
    _sync_teacher_from_student(teacher.module, model)
    # 统一设置动态位置编码（与单图测试保持一致）
    _set_dynamic_pos_embed(model)
    _set_dyna_ = getattr(teacher, "module", None) or teacher
    _set_dynamic_pos_embed(_set_dyna_)
    
    # 创建损失函数和优化器
    criterion = cfg.criterion.to(device)
    
    # 检查 optimizer 是实例还是类
    if hasattr(cfg.optimizer, '__call__') and not isinstance(cfg.optimizer, torch.optim.Optimizer):
        # 是一个可调用对象（函数或类）
        optimizer = cfg.optimizer(params=model.parameters())
    else:
        # 已经是实例化的优化器
        # 需要重新创建并绑定到当前模型参数
        optimizer_class = type(cfg.optimizer)
        optimizer_state = cfg.optimizer.state_dict()
        
        # 获取优化器参数
        defaults = cfg.optimizer.defaults
        optimizer = optimizer_class(model.parameters(), **defaults)
        
        print(f"✅ 创建优化器: {optimizer_class.__name__}")
    
    # 创建数据加载器
    from engine.data.ssod_group_reader import SSODGroupReader
    
    labeled_loader = cfg.train_dataloader
    unlabeled_loader = cfg.unlabeled_dataloader
    val_loader = cfg.val_dataloader
    
    # 创建分组读取器（处理弱/强增强）
    group_reader = SSODGroupReader(labeled_loader, unlabeled_loader)
    group_reader.set_epoch(0)
    
    # 创建后处理器和评估器
    postprocessor = cfg.postprocessor
    evaluator = cfg.evaluator
    
    # 混合精度
    scaler = GradScaler() if args.use_amp else None
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 伪标签配置（简化版：仅使用固定阈值 0.3）
    pseudo_cfg = PseudoCfg(
        score_thresh=0.3,
        input_size=args.input_size,
        vis_dir=args.vis_dir if args.vis_dir else None
    )
    pseudo_labeler = PseudoLabeler(pseudo_cfg)
    
    print(f"📋 伪标签配置:")
    print(f"   固定阈值: 0.3")
    print(f"   可视化目录: {pseudo_cfg.vis_dir if pseudo_cfg.vis_dir else '未启用'}")
    
    print(f"\n📋 训练配置:")
    print(f"   Epochs: {args.epochs}")
    print(f"   基础 λ_u: {args.lambda_u}")
    print(f"   Warmup 迭代: {schedules.warmup_iters}")
    print(f"   EMA 目标动量: {schedules.ema_target}")
    print(f"   EMA Warmup: {schedules.ema_warmup}")
    print(f"\n")
    
    # ========== 训练循环 ==========
    best_ap = 0.0
    
    for epoch in range(args.epochs):
        # 设置 epoch（用于数据加载器随机种子）
        group_reader.set_epoch(epoch)
        
        # 训练
        train_stats = train_one_epoch(
            student=model,
            teacher=teacher,
            criterion=criterion,
            optimizer=optimizer,
            group_reader=group_reader,
            device=device,
            epoch=epoch,
            cfg=cfg,
            pseudo_labeler=pseudo_labeler,
            schedules=schedules,
            base_lambda_u=args.lambda_u,
            scaler=scaler,
            debug_compare_first_iter=getattr(args, "debug_compare_first_iter", False),
        )
        
        # 评估
        if (epoch + 1) % args.eval_freq == 0:
            val_stats = evaluate(
                model=model,
                val_loader=val_loader,
                postprocessor=postprocessor,
                evaluator=evaluator,
                device=device,
            )
            
            # 保存最佳模型
            if val_stats and val_stats[0] > best_ap:
                best_ap = val_stats[0]
                if dist_utils.is_main_process():
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'ema': teacher.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'ap': best_ap,
                        'best_ap': best_ap,
                    }, output_dir / 'best_checkpoint.pth')
                    print(f"💾 保存最佳模型 (AP: {best_ap:.4f})")
        
        # 保存检查点
        if (epoch + 1) % args.save_freq == 0 and dist_utils.is_main_process():
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ap': best_ap,
                'best_ap': best_ap,
            }, output_dir / f'checkpoint_epoch{epoch:03d}.pth')
    
    print(f"\n{'='*80}")
    print(f"🎉 训练完成！")
    print(f"   最佳 AP: {best_ap:.4f}")
    print(f"{'='*80}\n")


# ========================================
# 入口
# ========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument('--config', type=str, 
                       default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth',
                       help='预训练检查点路径')
    parser.add_argument('--output-dir', type=str,
                       default='outputs/ssod_simple',
                       help='输出目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lambda-u', type=float, default=2.0,
                       help='无监督损失权重')
    parser.add_argument('--ema-momentum', type=float, default=0.999,
                       help='EMA 目标动量')
    parser.add_argument('--ema-warmup', type=int, default=100,
                       help='EMA 动量 warmup 步数')
    parser.add_argument('--warmup-iters', type=int, default=12000,
                       help='无监督损失 warmup 迭代数')
    parser.add_argument('--eval-freq', type=int, default=5,
                       help='评估频率（每N个epoch）')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='保存频率（每N个epoch）')
    
    # 伪标签参数
    parser.add_argument('--input-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--vis-dir', type=str, default='visualizations/teacher_pseudo',
                       help='教师伪标签可视化目录（设为None禁用）')
    
    # 设备参数
    parser.add_argument('--gpus', type=str, default='1,2',
                       help='GPU IDs，逗号分隔')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='使用混合精度训练')
    parser.add_argument('--seed', type=int, default=0,
                       help='随机种子')
    parser.add_argument('--debug-compare-first-iter', action='store_true', default=True,
                       help='首个iter进行师生同输入同分支等价性对比与差异打印（调试用）')
    
    args = parser.parse_args()
    
    # 解析 GPU 数量
    args.num_gpus = len(args.gpus.split(','))
    
    # 设置环境
    setup_environment(args.gpus)
    
    # 启动训练
    if args.num_gpus > 1:
        print(f"🚀 启动多进程训练，使用 {args.num_gpus} 个 GPU")
        mp.spawn(main, args=(args,), nprocs=args.num_gpus, join=True)
    else:
        print(f"🚀 启动单进程训练")
        main(0, args)
