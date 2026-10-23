"""
监督学习损失计算模块
用于半监督学习中的有标签数据监督训练
"""

import math
import torch
import torch.amp
from typing import Dict, List, Tuple

from ..misc import dist_utils
from .visualize_features import visualize_dinov3_features_simple


def _to_device_obj(obj, device):
    """Safely move tensors/tv_tensors recursively to device; keep strings/ints as-is."""
    import torch
    if isinstance(obj, dict):
        return {k: _to_device_obj(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ _to_device_obj(v, device) for v in obj ]
    # BoundingBoxes/Mask/Image also implement .to
    if hasattr(obj, 'to'):
        try:
            return obj.to(device)
        except Exception:
            return obj
    return obj


def compute_supervised_loss(model: torch.nn.Module,
                           criterion: torch.nn.Module,
                           images: torch.Tensor,
                           targets: List[Dict],
                           device: torch.device,
                           epoch: int,
                           step: int,
                           global_step: int,
                           epoch_step: int,
                           scaler=None) -> Tuple[torch.Tensor, Dict]:
    """
    计算监督学习损失
    
    Args:
        model: 学生模型
        criterion: 损失函数
        images: 输入图像 [B, C, H, W]
        targets: 目标标签列表
        device: 设备
        epoch: 当前epoch
        step: 当前step
        global_step: 全局step
        epoch_step: epoch总step数
        scaler: 混合精度scaler
        
    Returns:
        loss: 总损失
        loss_dict: 损失字典
    """
    # 移动到设备
    images = images.to(device)
    targets = [_to_device_obj(t, device) for t in targets]
    
    # 元数据
    metas = dict(epoch=epoch, step=step, global_step=global_step, epoch_step=epoch_step)
    
    # 🔥 确保模型在训练模式
    model.train()
    
    if scaler is not None:
        with torch.amp.autocast(device_type='cuda', enabled=True):
            outputs = model(images, targets=targets)
                      
        # NaN检查
        if 'pred_boxes' in outputs and torch.isnan(outputs['pred_boxes']).any():
            print("Warning: NaN detected in pred_boxes")
            
        with torch.amp.autocast(device_type='cuda', enabled=False):
            loss_dict = criterion(outputs, targets, **metas)
            
        loss = sum(loss_dict.values())
        
    else:
        outputs = model(images, targets=targets)
        loss_dict = criterion(outputs, targets, **metas)
        loss = sum(loss_dict.values())
    
    # 损失有效性检查
    if not math.isfinite(loss):
        print("Loss is {}, stopping training".format(loss))
        print(loss_dict)
        raise ValueError("Non-finite loss detected")
    
    return loss, loss_dict


def compute_supervised_loss_with_grad(model: torch.nn.Module,
                                    criterion: torch.nn.Module,
                                    images: torch.Tensor,
                                    targets: List[Dict],
                                    device: torch.device,
                                    epoch: int,
                                    step: int,
                                    global_step: int,
                                    epoch_step: int,
                                    optimizer: torch.optim.Optimizer,
                                    scaler=None,
                                    max_norm: float = 0) -> Tuple[torch.Tensor, Dict]:
    """
    计算监督学习损失并执行反向传播
    
    Args:
        model: 学生模型
        criterion: 损失函数
        images: 输入图像 [B, C, H, W]
        targets: 目标标签列表
        device: 设备
        epoch: 当前epoch
        step: 当前step
        global_step: 全局step
        epoch_step: epoch总step数
        optimizer: 优化器
        scaler: 混合精度scaler
        max_norm: 梯度裁剪阈值
        
    Returns:
        loss: 总损失
        loss_dict: 损失字典
    """
    loss, loss_dict = compute_supervised_loss(
        model, criterion, images, targets, device, 
        epoch, step, global_step, epoch_step, scaler
    )
    
    if scaler is not None:
        scaler.scale(loss).backward()
        
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    else:
        optimizer.zero_grad()
        loss.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
        optimizer.step()
    
    return loss, loss_dict


def reduce_supervised_loss(loss_dict: Dict) -> Dict:
    """
    分布式训练中减少损失字典
    
    Args:
        loss_dict: 损失字典
        
    Returns:
        减少后的损失字典
    """
    return dist_utils.reduce_dict(loss_dict)
