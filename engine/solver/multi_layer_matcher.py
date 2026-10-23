"""
多层匹配模块

本模块提供封装好的多层匹配函数，参考 deim_criterion.py 的实现，
支持主输出、辅助层输出、编码器输出的统一匹配，并返回代价矩阵。
"""

import torch
from typing import Dict, List, Tuple, Optional, Any


__all__ = ['compute_multi_layer_matching']


def compute_multi_layer_matching(
    outputs: Dict[str, Any],
    targets: List[Dict[str, torch.Tensor]],
    matcher,
    epoch: int = 0,
    return_cost: bool = True,
    compute_union_set: bool = True,
):
    """
    计算多层匹配（主输出 + 辅助层 + 编码器），并返回代价矩阵和统一匹配集合
    
    Args:
        outputs: dict, 模型输出，包含：
            - 'pred_logits': [B, N, num_classes] 主输出分类logits
            - 'pred_boxes': [B, N, 4] 主输出边界框
            - 'aux_outputs': List[dict], 辅助层输出（解码器中间层）
            - 'pre_outputs': dict, 传统头输出（第一层解码器）
            - 'enc_aux_outputs': List[dict], 编码器辅助输出
        
        targets: List[dict], 每个样本的目标，包含：
            - 'labels': [N_gt] 类别标签
            - 'boxes': [N_gt, 4] 归一化边界框 (cxcywh)
        
        matcher: HungarianMatcher, 匹配器实例
        
        epoch: int, 当前训练轮数（用于 warm-up 判断）
        
        return_cost: bool, 是否返回代价矩阵
        
        compute_union_set: bool, 是否计算统一匹配集合
    
    Returns:
        result: dict, 包含：
            - 'indices': List[Tuple], 主输出的匹配索引 (one2one)
            - 'indices_o2o': List[Tuple], 显式的 one2one 索引（与 indices 相同）
            - 'cost_matrix': Tensor, 代价矩阵 [bs, num_queries, sum(sizes)]
            - 'sizes': List[int], 每个样本的 GT 数量
            - 'matched_costs_for_gmm': List[Tensor], 用于 GMM 的匹配代价
            - 'cached_indices': List[List[Tuple]], 每个辅助层的匹配索引
            - 'cached_indices_enc': List[List[Tuple]], 每个编码器层的匹配索引
            - 'indices_go': List[Tuple], 统一匹配集合（跨层）
            - 'num_boxes_go': float, 统一集合的归一化因子
    """
    
    # 提取不含辅助输出的主输出
    outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}
    
    # ========== 1. 主输出匹配 ==========
    match_results = matcher(
        outputs_without_aux, 
        targets, 
        epoch=epoch,
        return_cost=return_cost
    )
    
    # 清理缓存
    if hasattr(matcher, '_clear_cache'):
        matcher._clear_cache()
    
    # 提取主输出的匹配索引
    indices = match_results['indices']
    
    # 初始化返回结果
    result = {
        'indices': indices,
        'indices_o2o': indices,  # 显式标记为 one2one
    }
    
    # 如果需要返回代价矩阵，添加到结果中
    if return_cost and 'cost_matrix' in match_results:
        result['cost_matrix'] = match_results['cost_matrix']
        result['sizes'] = match_results['sizes']
        result['matched_costs_for_gmm'] = match_results.get('matched_costs_for_gmm', [])
    
    # ========== 2. 辅助层和编码器层匹配（如果需要计算统一集合）==========
    if compute_union_set and 'aux_outputs' in outputs:
        indices_aux_list = []
        cached_indices = []
        cached_indices_enc = []
        
        # 2.1 辅助层匹配（解码器中间层）
        aux_outputs_list = outputs['aux_outputs']
        if 'pre_outputs' in outputs:
            # 传统头（pre）也作为一个"辅助层"加入统一集合
            aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
        
        for i, aux_outputs in enumerate(aux_outputs_list):
            # 对每个 decoder 辅助层做匹配
            aux_match = matcher(aux_outputs, targets, epoch=epoch)
            indices_aux = aux_match['indices']
            cached_indices.append(indices_aux)
            indices_aux_list.append(indices_aux)
            
            # 清理缓存
            if hasattr(matcher, '_clear_cache'):
                matcher._clear_cache()
        
        # 2.2 编码器层匹配
        if 'enc_aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                # 对每个 encoder 辅助层做匹配
                enc_match = matcher(aux_outputs, targets, epoch=epoch)
                indices_enc = enc_match['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
                
                # 清理缓存
                if hasattr(matcher, '_clear_cache'):
                    matcher._clear_cache()
        
        # 2.3 计算统一匹配集合（跨所有层的匹配联合）
        indices_go = _get_go_indices(indices, indices_aux_list)
        
        # 2.4 计算统一集合的归一化因子
        num_boxes_go = sum(len(x[0]) for x in indices_go)
        num_boxes_go = torch.as_tensor(
            [num_boxes_go], 
            dtype=torch.float, 
            device=next(iter(outputs.values())).device
        )
        
        # 分布式同步
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes_go)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        
        num_boxes_go = torch.clamp(num_boxes_go / world_size, min=1).item()
        
        # 添加到返回结果
        result['cached_indices'] = cached_indices
        result['cached_indices_enc'] = cached_indices_enc
        result['indices_go'] = indices_go
        result['num_boxes_go'] = num_boxes_go
    
    return result


def _get_go_indices(
    indices: List[Tuple[torch.Tensor, torch.Tensor]], 
    indices_aux_list: List[List[Tuple[torch.Tensor, torch.Tensor]]]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    计算跨所有层的匹配联合集合（去重）
    
    这个函数将主输出的匹配索引与所有辅助层的匹配索引合并，
    对于同一个 query 匹配到多个 GT 的情况，保留出现次数最多的匹配。
    
    Args:
        indices: List[Tuple], 主输出的匹配索引
        indices_aux_list: List[List[Tuple]], 所有辅助层的匹配索引列表
    
    Returns:
        results: List[Tuple], 统一后的匹配索引（每个样本一对）
    """
    # 将主输出索引与所有辅助层索引合并
    for indices_aux in indices_aux_list:
        indices = [
            (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
            for idx1, idx2 in zip(indices, indices_aux)
        ]
    
    # 对每个样本的匹配进行去重（保留出现次数最多的匹配）
    results = []
    for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
        # ind: [N, 2], 每行是 (query_idx, gt_idx)
        
        # 统计每个 (query, gt) 对的出现次数
        unique, counts = torch.unique(ind, return_counts=True, dim=0)
        
        # 按出现次数降序排序
        count_sort_indices = torch.argsort(counts, descending=True)
        unique_sorted = unique[count_sort_indices]
        
        # 对每个 query，只保留出现次数最多的 gt
        column_to_row = {}
        for idx in unique_sorted:
            row_idx, col_idx = idx[0].item(), idx[1].item()
            if row_idx not in column_to_row:
                column_to_row[row_idx] = col_idx
        
        # 转换回 (query_indices, gt_indices) 格式
        final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
        final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
        results.append((final_rows.long(), final_cols.long()))
    
    return results


def extract_one2one_costs_from_matrix(
    cost_matrix: torch.Tensor,
    sizes: List[int],
    indices_o2o: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[torch.Tensor]:
    """
    从完整代价矩阵中提取 one2one 匹配的代价
    
    Args:
        cost_matrix: [bs, num_queries, sum(sizes)], 完整代价矩阵
        sizes: List[int], 每个样本的 GT 数量
        indices_o2o: List[Tuple], one2one 匹配索引
    
    Returns:
        match_costs: List[Tensor], 每个样本的匹配代价 [N_matched]
    """
    C_split = cost_matrix.split(sizes, dim=-1)
    
    match_costs = []
    for img_id, (query_idx, gt_idx) in enumerate(indices_o2o):
        if len(query_idx) == 0:
            match_costs.append(cost_matrix.new_zeros(0))
        else:
            # 提取 one2one 匹配的代价
            cost_per_img = C_split[img_id][img_id, query_idx, gt_idx]
            match_costs.append(cost_per_img)
    
    return match_costs

