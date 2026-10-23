"""
无监督损失计算辅助模块

本模块提供了封装好的无监督损失计算函数，参考 DEIMCriterion 的实现，
支持主输出、辅助层输出、编码器输出、DN查询输出等多种损失计算。
"""

import torch
import torch.nn as nn
from ..deim.deim_criterion import DEIMCriterion
from ..deim.matcher import HungarianMatcher

__all__ = ['compute_unsup_losses']


def compute_unsup_losses(
    outputs,
    targets,
    matcher,
    criterion,
    ssl_lambda_u=1.0,
    use_uni_set=True,
    prefix='unsup_',
):
    """
    计算无监督损失（支持主输出 + 辅助层 + 编码器 + DN分支）
    
    Args:
        outputs: dict, 模型输出，包含：
            - 'pred_logits': [B, N, num_classes] 主输出分类logits
            - 'pred_boxes': [B, N, 4] 主输出边界框
            - 'pred_corners': [B, N, ...] 主输出角点（FGL损失用）
            - 'ref_points': [B, N, 4] 主输出参考点
            - 'up': int, 上采样因子
            - 'reg_scale': float, 回归缩放因子
            - 'aux_outputs': List[dict], 辅助层输出（解码器中间层）
            - 'pre_outputs': dict, 传统头输出（第一层解码器）
            - 'enc_aux_outputs': List[dict], 编码器辅助输出
            - 'enc_meta': dict, 编码器元信息
            - 'dn_outputs': List[dict], DN查询输出
            - 'dn_pre_outputs': dict, DN传统头输出
            - 'dn_meta': dict, DN元信息
        
        targets: List[dict], 每个样本的目标，包含：
            - 'labels': [N_gt] 类别标签
            - 'boxes': [N_gt, 4] 归一化边界框 (cxcywh)
        
        matcher: HungarianMatcher, 匹配器实例
        
        criterion: DEIMCriterion, 损失计算器实例
        
        ssl_lambda_u: float, 半监督损失权重系数
        
        use_uni_set: bool, 是否对定位损失使用统一匹配集合
        
        prefix: str, 损失名称前缀（默认 'unsup_'）
    
    Returns:
        losses: dict, 所有损失项的字典
    """
    
    # 提取不含辅助输出的主输出
    outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
    
    # ========== 1. 主输出匹配 ==========
    indices = matcher(outputs_without_aux, targets)['indices']
    
    # 清理缓存
    if hasattr(matcher, '_clear_cache'):
        matcher._clear_cache()
    
    # ========== 2. 计算统一匹配集合（如果需要）==========
    indices_go = None
    num_boxes_go = None
    cached_indices = []
    cached_indices_enc = []
    
    if use_uni_set and 'aux_outputs' in outputs:
        indices_aux_list = []
        
        # 辅助层匹配
        aux_outputs_list = outputs['aux_outputs']
        if 'pre_outputs' in outputs:
            aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
        
        for aux_outputs in aux_outputs_list:
            indices_aux = matcher(aux_outputs, targets)['indices']
            cached_indices.append(indices_aux)
            indices_aux_list.append(indices_aux)
        
        # 编码器层匹配
        if 'enc_aux_outputs' in outputs:
            for aux_outputs in outputs['enc_aux_outputs']:
                indices_enc = matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
        
        # 计算统一集合
        indices_go = criterion._get_go_indices(indices, indices_aux_list)
        
        # 统一集合的归一化因子
        num_boxes_go = sum(len(x[0]) for x in indices_go)
        num_boxes_go = torch.as_tensor(
            [num_boxes_go], 
            dtype=torch.float, 
            device=next(iter(outputs.values())).device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes_go)
        num_boxes_go = torch.clamp(num_boxes_go / max(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1, 1), min=1).item()
    
    # ========== 3. 计算常规归一化因子 ==========
    num_boxes = sum(len(t["labels"]) for t in targets)
    num_boxes = torch.as_tensor(
        [num_boxes], 
        dtype=torch.float, 
        device=next(iter(outputs.values())).device
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(num_boxes)
    num_boxes = torch.clamp(num_boxes / max(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1, 1), min=1).item()
    
    # ========== 4. 计算主输出损失 ==========
    losses = {}
    for loss in criterion.losses:
        use_uni = use_uni_set and (loss in ['boxes', 'local'])
        indices_in = indices_go if use_uni else indices
        num_boxes_in = num_boxes_go if use_uni else num_boxes
        
        meta = criterion.get_loss_meta_info(loss, outputs, targets, indices_in)
        l_dict = criterion.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
        l_dict = {k: l_dict[k] * criterion.weight_dict[k] for k in l_dict if k in criterion.weight_dict}
        l_dict = {k: v * float(ssl_lambda_u) for k, v in l_dict.items()}
        l_dict = {prefix + k: v for k, v in l_dict.items()}
        losses.update(l_dict)
    
    # ========== 5. 计算辅助层损失 ==========
    if 'aux_outputs' in outputs:
        for i, aux_outputs in enumerate(outputs['aux_outputs']):
            if 'local' in criterion.losses:
                aux_outputs['up'] = outputs['up']
                aux_outputs['reg_scale'] = outputs['reg_scale']
            
            for loss in criterion.losses:
                use_uni = use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni else (cached_indices[i] if cached_indices else indices)
                num_boxes_in = num_boxes_go if use_uni else num_boxes
                
                meta = criterion.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = criterion.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * criterion.weight_dict[k] for k in l_dict if k in criterion.weight_dict}
                l_dict = {k: v * float(ssl_lambda_u) for k, v in l_dict.items()}
                l_dict = {prefix + k + f'_aux_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    
    # ========== 6. 计算传统头损失（pre_outputs）==========
    if 'pre_outputs' in outputs:
        aux_outputs = outputs['pre_outputs']
        for loss in criterion.losses:
            use_uni = use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni else (cached_indices[-1] if cached_indices else indices)
            num_boxes_in = num_boxes_go if use_uni else num_boxes
            
            meta = criterion.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
            l_dict = criterion.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * criterion.weight_dict[k] for k in l_dict if k in criterion.weight_dict}
            l_dict = {k: v * float(ssl_lambda_u) for k, v in l_dict.items()}
            l_dict = {prefix + k + '_pre': v for k, v in l_dict.items()}
            losses.update(l_dict)
    
    # ========== 7. 计算编码器辅助损失 ==========
    if 'enc_aux_outputs' in outputs:
        assert 'enc_meta' in outputs, 'enc_meta required for encoder auxiliary outputs'
        class_agnostic = outputs['enc_meta']['class_agnostic']
        
        if class_agnostic:
            import copy
            orig_num_classes = criterion.num_classes
            criterion.num_classes = 1
            enc_targets = copy.deepcopy(targets)
            for t in enc_targets:
                t['labels'] = torch.zeros_like(t["labels"])
        else:
            enc_targets = targets
        
        for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
            for loss in criterion.losses:
                use_uni = use_uni_set and (loss == 'boxes')
                indices_in = indices_go if use_uni else (cached_indices_enc[i] if cached_indices_enc else indices)
                num_boxes_in = num_boxes_go if use_uni else num_boxes
                
                meta = criterion.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                l_dict = criterion.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * criterion.weight_dict[k] for k in l_dict if k in criterion.weight_dict}
                l_dict = {k: v * float(ssl_lambda_u) for k, v in l_dict.items()}
                l_dict = {prefix + k + f'_enc_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        if class_agnostic:
            criterion.num_classes = orig_num_classes
    
    # ========== 8. 计算DN查询损失 ==========
    if 'dn_outputs' in outputs:
        assert 'dn_meta' in outputs, 'dn_meta required for DN outputs'
        indices_dn = criterion.get_cdn_matched_indices(outputs['dn_meta'], targets)
        dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
        
        for i, aux_outputs in enumerate(outputs['dn_outputs']):
            if 'local' in criterion.losses:
                aux_outputs['is_dn'] = True
                aux_outputs['up'] = outputs['up']
                aux_outputs['reg_scale'] = outputs['reg_scale']
            
            for loss in criterion.losses:
                meta = criterion.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                l_dict = criterion.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                l_dict = {k: l_dict[k] * criterion.weight_dict[k] for k in l_dict if k in criterion.weight_dict}
                l_dict = {k: v * float(ssl_lambda_u) for k, v in l_dict.items()}
                l_dict = {prefix + k + f'_dn_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        
        # DN传统头损失
        if 'dn_pre_outputs' in outputs:
            aux_outputs = outputs['dn_pre_outputs']
            for loss in criterion.losses:
                meta = criterion.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                l_dict = criterion.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                l_dict = {k: l_dict[k] * criterion.weight_dict[k] for k in l_dict if k in criterion.weight_dict}
                l_dict = {k: v * float(ssl_lambda_u) for k, v in l_dict.items()}
                l_dict = {prefix + k + '_dn_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)
    
    # ========== 9. 数值稳定化 ==========
    losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
    
    return losses

