"""
无监督检测损失（unsup_loss）计算模块

本文件聚焦于半监督/教师-学生框架下的无监督损失构建：
- 输入：学生/教师伪标签、坐标对齐后的伪框/类别/分数、以及模型的解码输出
- 核心：基于分类/回归/GIoU 等代价的匈牙利一对一分配（或后续可扩展的一对多策略）
- 目标：返回用于反向传播的无监督损失项（或其中间匹配与筛选结果）

注意：类接口使用 forward(...) 以便与 nn.Module/注册系统对接；若需函数式调用，可在本文件末尾使用包装函数。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..deim.deim_criterion import DEIMCriterion
from ..deim.matcher import HungarianMatcher

from ..core import register


def _get_model(model):
    """获取原始模型，兼容 DDP 包装"""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model



@register()
class UnsupLossComputer(nn.Module):
    """无监督损失计算器（基于匈牙利/代价的一对一匹配）。

    说明：传统 DETR 风格的无目标类别不参与 targets；通常预测数 > 目标数，因此采用 1-to-1 最优分配，
    其余未匹配的预测视为背景。
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                student: Optional[Any] = None,
                teacher: Optional[Any] = None,
                ssl_lambda_u: float = 1.0,
                ):
        """
        简化版无监督损失计算器
        
        Args:
            weight_dict: 匹配代价权重字典
            use_focal_loss: 是否使用 Focal Loss
            alpha: Focal Loss alpha 参数
            gamma: Focal Loss gamma 参数
            student: 学生模型
            teacher: 教师模型
            ssl_lambda_u: 无监督损失权重（保留兼容性，未使用）
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

        # 挂载学生与教师模型（用于提取 num_classes 等信息）
        self.student = student
        self.teacher = teacher

    def _build_matcher(self):
        """统一构建 HungarianMatcher，便于多处复用"""
        return HungarianMatcher(
            weight_dict={
                'cost_class': self.cost_class,
                'cost_bbox': self.cost_bbox,
                'cost_giou': self.cost_giou
            },
            use_focal_loss=self.use_focal_loss,
            alpha=self.alpha,
            gamma=self.gamma,
        )

    def _ensure_matcher(self):
        """确保 matcher 存在"""
        if getattr(self, 'matcher', None) is None:
            self.matcher = self._build_matcher()
        return self.matcher

    def forward(self, student_pseudo, teacher_pseudo, pseudo_bboxes, pseudo_labels, pseudo_scores, epoch=0, **kwargs):
        """
        简化版无监督损失计算：参考 supervised_loss.py 的方式
        
        Args:
            student_pseudo: 学生模型的完整输出（包含 pred_logits, pred_boxes, aux_outputs 等）
            teacher_pseudo: 教师模型的伪标签（未使用，保留接口兼容性）
            pseudo_bboxes: 教师筛选后的伪标签边界框列表 [List[Tensor]]
            pseudo_labels: 教师筛选后的伪标签类别列表 [List[Tensor]]
            pseudo_scores: 教师筛选后的伪标签置信度列表 [List[Tensor]]
            epoch: 当前 epoch
        """
        # 构建 targets 格式（与监督学习一致）
        targets = []
        for boxes, labels in zip(pseudo_bboxes, pseudo_labels):
            # 清理框，防止 NaN/Inf
            boxes_clean = torch.nan_to_num(boxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
            boxes_clean[..., 2:] = boxes_clean[..., 2:].clamp(min=1e-6)
            
            targets.append({
                'boxes': boxes_clean,
                'labels': labels,
            })
        
        # 准备学生输出（使用 student_pseudo 中已有的完整输出）
        outputs = {
            'pred_logits': student_pseudo['pred_logits'],
            'pred_boxes': student_pseudo['pred_boxes'],
        }
        
        # 添加辅助层输出（如果存在）
        if 'aux_outputs' in student_pseudo:
            outputs['aux_outputs'] = student_pseudo['aux_outputs']
        if 'enc_aux_outputs' in student_pseudo:
            outputs['enc_aux_outputs'] = student_pseudo['enc_aux_outputs']
        if 'pre_outputs' in student_pseudo:
            outputs['pre_outputs'] = student_pseudo['pre_outputs']
        if 'dn_outputs' in student_pseudo:
            outputs['dn_outputs'] = student_pseudo['dn_outputs']
        if 'dn_pre_outputs' in student_pseudo:
            outputs['dn_pre_outputs'] = student_pseudo['dn_pre_outputs']
        if 'dn_meta' in student_pseudo:
            outputs['dn_meta'] = student_pseudo['dn_meta']
        if 'enc_meta' in student_pseudo:
            outputs['enc_meta'] = student_pseudo['enc_meta']
        if 'up' in student_pseudo:
            outputs['up'] = student_pseudo['up']
        if 'reg_scale' in student_pseudo:
            outputs['reg_scale'] = student_pseudo['reg_scale']
        
        # 获取或创建 matcher
        matcher = self._ensure_matcher()
        
        # 获取学生模型，用于提取 num_classes 等信息
        student_model = _get_model(self.student) if hasattr(self, 'student') else None
        num_classes = outputs['pred_logits'].shape[-1]
        reg_max = getattr(student_model.decoder, 'reg_max', 32) if student_model and hasattr(student_model, 'decoder') else 32
        
        # 组装 criterion（使用与监督学习相同的损失函数）
        criterion = DEIMCriterion(
            matcher=matcher,
            weight_dict={
                'loss_mal': 2.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0,
                'loss_fgl': 1.0,
            },
            losses=['mal', 'boxes', 'local'],
            alpha=self.alpha,
            gamma=self.gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=None,
            use_uni_set=True,
        )

        # 计算损失（与监督学习相同的方式）
        metas = dict(epoch=epoch)
        loss_dict = criterion(outputs, targets, **metas)
        
        # 释放 criterion
        del criterion
        
        # ========== 2. 计算教师-学生特征一致性损失 ==========
        # 理念：同一张图片经过弱增强（教师）和强增强（学生）后，
        #       模型提取的特征应该在语义上保持一致，这是一致性正则化的核心
        if hasattr(self, 'teacher') and self.teacher is not None:
            teacher_model = _get_model(self.teacher)
            images_strong = student_pseudo['image']  # 学生的 strong image
            
            # 教师在强增强图像上前向传播（不需要梯度）
            with torch.no_grad():
                teacher_outputs = teacher_model(images_strong, is_teacher=True)
            
            # 提取学生和教师的 decoder 特征（不再提取encoder特征）
            student_decoder_feats = student_pseudo.get('dec_hs')  # List[Tensor], 每层 decoder 的隐藏状态
            teacher_decoder_feats = teacher_outputs.get('dec_hs')  # List[Tensor]
            
            consistency_loss = 0.0
            consistency_count = 0
            
            if student_decoder_feats is not None and teacher_decoder_feats is not None:
                if isinstance(student_decoder_feats, (list, tuple)) and isinstance(teacher_decoder_feats, (list, tuple)):
                    num_layers = min(len(student_decoder_feats), len(teacher_decoder_feats))
                    for layer in range(num_layers):
                        s_dec = student_decoder_feats[layer]
                        t_dec = teacher_decoder_feats[layer]
                        
                        if s_dec is not None and t_dec is not None:
                            # 清理 NaN/Inf
                            s_dec = torch.nan_to_num(s_dec, nan=0.0, posinf=1e4, neginf=-1e4)
                            t_dec = torch.nan_to_num(t_dec, nan=0.0, posinf=1e4, neginf=-1e4)
                            
                            # 计算 MSE 一致性损失
                            dec_loss = torch.nn.functional.mse_loss(s_dec, t_dec.detach())
                            loss_dict[f'consistency_decoder_layer{layer}_unsup'] = dec_loss
                            consistency_loss += dec_loss
                            consistency_count += 1
            
            # 如果有一致性损失，添加到 loss_dict
            if consistency_count > 0:
                loss_dict['consistency_total_unsup'] = consistency_loss / consistency_count
        
        return loss_dict

 
