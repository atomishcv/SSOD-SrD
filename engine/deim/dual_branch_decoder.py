"""
Dual-Branch DEIM Decoder for Multi-Granularity Object Detection
基于DEIMv2的双分支解码器，用于同时检测单株（细粒度）和群落（粗粒度）目标

核心设计思路：
1. 多尺度特征融合：
   - 单株分支：融合高分辨率特征（level 0 和 level 1）
   - 集群分支：融合低分辨率特征（level 1 和 level 2）
   - 中间尺度（level 1）作为共享桥梁
2. 尺度引导的Query设计：不同分支使用不同尺度先验的查询
3. 独立的解码器和检测头

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import math
import copy
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..core import register
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid, bias_init_with_prob
from .dfine_decoder import MSDeformableAttention, LQE, Integral
from .dfine_utils import weighting_function, distance2bbox
from .deim_utils import RMSNorm, SwiGLUFFN, Gate, MLP

__all__ = ['DualBranchDEIMTransformer', 'MultiScaleFusionModule', 'ScaleAwareQueryGenerator']


# ============================================================================
# 多尺度特征融合模块 (Multi-Scale Fusion Module)
# ============================================================================
class MultiScaleFusionModule(nn.Module):
    """
    多尺度特征融合模块：
    - 单株分支：融合 level 0 (高分辨率) 和 level 1 (中分辨率)
    - 集群分支：融合 level 1 (中分辨率) 和 level 2 (低分辨率)
    
    使用 Cross-Attention 进行跨尺度融合
    """
    
    def __init__(self, 
                 hidden_dim: int = 256, 
                 nhead: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        
        # 单株分支融合：level 0 + level 1 (高分辨率特征)
        self.individual_fusion = ScaleFusionBlock(
            hidden_dim, nhead, dropout, 
            name='individual',
            upsample_low=True  # 将低分辨率上采样到高分辨率
        )
        
        # 集群分支融合：level 1 + level 2 (低分辨率特征)
        self.colony_fusion = ScaleFusionBlock(
            hidden_dim, nhead, dropout,
            name='colony',
            upsample_low=False  # 将高分辨率下采样到低分辨率
        )
        
        # 可选：综合融合模块（融合所有三个尺度）
        self.combined_fusion = CombinedScaleFusion(hidden_dim, nhead, dropout)
        
    def forward(self, proj_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            proj_feats: 投影后的多尺度特征列表
                       [feat_0: [B, C, H0, W0], feat_1: [B, C, H1, W1], feat_2: [B, C, H2, W2]]
        
        Returns:
            Dict containing:
                - 'individual': 单株分支融合特征 [B, H0*W0 + H1*W1, C]
                - 'colony': 集群分支融合特征 [B, H1*W1 + H2*W2, C]
                - 'combined': 综合融合特征 [B, sum(Hi*Wi), C]
                - 'spatial_shapes_individual': 单株分支空间尺寸
                - 'spatial_shapes_colony': 集群分支空间尺寸
        """
        assert len(proj_feats) >= 3, "Need at least 3 scale features"
        
        feat_0, feat_1, feat_2 = proj_feats[0], proj_feats[1], proj_feats[2]
        
        # 单株分支：融合 level 0 和 level 1
        individual_memory, individual_shapes = self.individual_fusion(feat_0, feat_1)
        
        # 集群分支：融合 level 1 和 level 2
        colony_memory, colony_shapes = self.colony_fusion(feat_1, feat_2)
        
        # 综合融合：所有三个尺度
        combined_memory, combined_shapes = self.combined_fusion(proj_feats[:3])
        
        return {
            'individual': individual_memory,
            'colony': colony_memory,
            'combined': combined_memory,
            'spatial_shapes_individual': individual_shapes,
            'spatial_shapes_colony': colony_shapes,
            'spatial_shapes_combined': combined_shapes
        }


class ScaleFusionBlock(nn.Module):
    """
    两个尺度的特征融合块
    参考 deim_decoder.py 的 _get_encoder_input 方式：简单展平并拼接
    """
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 nhead: int = 8,
                 dropout: float = 0.0,
                 name: str = 'fusion',
                 upsample_low: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.name = name
        
    def forward(self, feat_high: torch.Tensor, feat_low: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        参考 _get_encoder_input 的方式：
        1. 展平各层空间维 [B, C, H, W] -> [B, H*W, C]
        2. 按层在长度维 concat 得到 [B, sum(H*W), C]
        3. 记录每层空间尺寸 [H, W]
        
        Args:
            feat_high: 高分辨率特征 [B, C, H_h, W_h]
            feat_low: 低分辨率特征 [B, C, H_l, W_l]
        
        Returns:
            fused_memory: 融合后的特征 [B, H_h*W_h + H_l*W_l, C]
            spatial_shapes: 各层空间尺寸 [[H_h, W_h], [H_l, W_l]]
        """
        feat_flatten = []
        spatial_shapes = []
        
        # 处理高分辨率特征
        _, _, h_h, w_h = feat_high.shape
        feat_flatten.append(feat_high.flatten(2).permute(0, 2, 1))  # [B, H_h*W_h, C]
        spatial_shapes.append([h_h, w_h])
        
        # 处理低分辨率特征
        _, _, h_l, w_l = feat_low.shape
        feat_flatten.append(feat_low.flatten(2).permute(0, 2, 1))  # [B, H_l*W_l, C]
        spatial_shapes.append([h_l, w_l])
        
        # 按层在长度维 concat 得到 [B, H_h*W_h + H_l*W_l, C]
        fused_memory = torch.cat(feat_flatten, dim=1)
        
        return fused_memory, spatial_shapes


class CombinedScaleFusion(nn.Module):
    """
    综合三个尺度的特征融合
    参考 deim_decoder.py 的 _get_encoder_input 方式
    """
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 nhead: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        参考 _get_encoder_input 的方式：
        1. 展平各层空间维 [B, C, H, W] -> [B, H*W, C]
        2. 按层在长度维 concat 得到 [B, sum(H*W), C]
        3. 记录每层空间尺寸 [H, W]
        
        Args:
            feats: 三个尺度的特征列表
                   [feat_0: [B, C, H0, W0], feat_1: [B, C, H1, W1], feat_2: [B, C, H2, W2]]
        
        Returns:
            feat_flatten: 展平并拼接后的特征 [B, sum(H*W), C]
            spatial_shapes: 各层空间尺寸列表 [[H0, W0], [H1, W1], [H2, W2]]
        """
        feat_flatten = []
        spatial_shapes = []
        
        for feat in feats:
            _, _, h, w = feat.shape
            # [B, C, H, W] -> [B, H*W, C]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # 保存每层的空间尺寸 [H, W]
            spatial_shapes.append([h, w])
        
        # 按层在长度维 concat 得到 [B, sum(H*W), C]
        feat_flatten = torch.cat(feat_flatten, dim=1)
        
        return feat_flatten, spatial_shapes


# ============================================================================
# 尺度感知查询生成器 (Scale-Aware Query Generator)
# ============================================================================
class ScaleAwareQueryGenerator(nn.Module):
    """
    尺度感知查询生成器：
    1. 为单株分支生成小尺度先验的查询（关注高分辨率特征）
    2. 为群落分支生成大尺度先验的查询（关注低分辨率特征）
    3. 为综合分支生成全尺度范围的查询（关注所有尺度特征）
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 num_queries_individual: int = 300,  # 单株分支查询数
                 num_queries_colony: int = 300,      # 群落分支查询数
                 num_queries_combined: int = 300,    # 综合分支查询数
                 scale_individual: Tuple[float, float] = (0.02, 0.15),  # 单株尺度范围
                 scale_colony: Tuple[float, float] = (0.1, 0.5),        # 群落尺度范围
                 scale_combined: Tuple[float, float] = (0.02, 0.5)):    # 综合尺度范围
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries_individual = num_queries_individual
        self.num_queries_colony = num_queries_colony
        self.num_queries_combined = num_queries_combined
        self.scale_individual = scale_individual
        self.scale_colony = scale_colony
        self.scale_combined = scale_combined
        
        # 可学习的查询内容嵌入
        self.query_embed_individual = nn.Embedding(num_queries_individual, hidden_dim)
        self.query_embed_colony = nn.Embedding(num_queries_colony, hidden_dim)
        self.query_embed_combined = nn.Embedding(num_queries_combined, hidden_dim)
        
        # 可学习的尺度嵌入（用于初始化锚框的w,h）
        self.scale_embed_individual = nn.Embedding(num_queries_individual, 2)  # w, h
        self.scale_embed_colony = nn.Embedding(num_queries_colony, 2)
        self.scale_embed_combined = nn.Embedding(num_queries_combined, 2)
        
        # 位置嵌入生成器
        self.pos_embed_individual = nn.Embedding(num_queries_individual, 2)  # x, y
        self.pos_embed_colony = nn.Embedding(num_queries_colony, 2)
        self.pos_embed_combined = nn.Embedding(num_queries_combined, 2)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # 初始化查询嵌入
        init.xavier_uniform_(self.query_embed_individual.weight)
        init.xavier_uniform_(self.query_embed_colony.weight)
        init.xavier_uniform_(self.query_embed_combined.weight)
        
        # 初始化尺度嵌入：单株偏小，群落偏大，综合覆盖全范围
        scale_ind_min, scale_ind_max = self.scale_individual
        scale_col_min, scale_col_max = self.scale_colony
        scale_comb_min, scale_comb_max = self.scale_combined
        
        # 单株尺度初始化
        scale_ind = torch.rand(self.num_queries_individual, 2) * (scale_ind_max - scale_ind_min) + scale_ind_min
        self.scale_embed_individual.weight.data = scale_ind
        
        # 群落尺度初始化
        scale_col = torch.rand(self.num_queries_colony, 2) * (scale_col_max - scale_col_min) + scale_col_min
        self.scale_embed_colony.weight.data = scale_col
        
        # 综合尺度初始化（覆盖全范围）
        scale_comb = torch.rand(self.num_queries_combined, 2) * (scale_comb_max - scale_comb_min) + scale_comb_min
        self.scale_embed_combined.weight.data = scale_comb
        
        # 位置初始化：均匀分布
        init.uniform_(self.pos_embed_individual.weight, 0.1, 0.9)
        init.uniform_(self.pos_embed_colony.weight, 0.1, 0.9)
        init.uniform_(self.pos_embed_combined.weight, 0.1, 0.9)
    
    def forward(self, batch_size: int, device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        生成三个分支的查询
        """
        # 单株分支查询
        query_content_ind = self.query_embed_individual.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos_ind = self.pos_embed_individual.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_scale_ind = self.scale_embed_individual.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 群落分支查询
        query_content_col = self.query_embed_colony.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos_col = self.pos_embed_colony.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_scale_col = self.scale_embed_colony.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 综合分支查询
        query_content_comb = self.query_embed_combined.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_pos_comb = self.pos_embed_combined.weight.unsqueeze(0).expand(batch_size, -1, -1)
        query_scale_comb = self.scale_embed_combined.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 构建锚框 (x, y, w, h)
        anchor_ind = torch.cat([
            torch.sigmoid(query_pos_ind),  # x, y in [0, 1]
            torch.sigmoid(query_scale_ind)  # w, h in [0, 1]
        ], dim=-1)
        
        anchor_col = torch.cat([
            torch.sigmoid(query_pos_col),
            torch.sigmoid(query_scale_col)
        ], dim=-1)
        
        anchor_comb = torch.cat([
            torch.sigmoid(query_pos_comb),
            torch.sigmoid(query_scale_comb)
        ], dim=-1)
        
        # 转换为未激活形式（用于后续refine）
        anchor_ind_unact = inverse_sigmoid(anchor_ind.clamp(min=1e-4, max=1-1e-4))
        anchor_col_unact = inverse_sigmoid(anchor_col.clamp(min=1e-4, max=1-1e-4))
        anchor_comb_unact = inverse_sigmoid(anchor_comb.clamp(min=1e-4, max=1-1e-4))
        
        return {
            'individual': {
                'content': query_content_ind,
                'anchor_unact': anchor_ind_unact,
                'anchor': anchor_ind
            },
            'colony': {
                'content': query_content_col,
                'anchor_unact': anchor_col_unact,
                'anchor': anchor_col
            },
            'combined': {
                'content': query_content_comb,
                'anchor_unact': anchor_comb_unact,
                'anchor': anchor_comb
            }
        }


# ============================================================================
# 分支解码器层 (Branch Decoder Layer)
# ============================================================================
class BranchDecoderLayer(nn.Module):
    """
    单个分支的解码器层
    """
    
    def __init__(self,
                 d_model: int = 256,
                 n_head: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.,
                 activation: str = 'relu',
                 n_levels: int = 2,  # 每个分支只关注2个尺度
                 n_points: int = 4,
                 cross_attn_method: str = 'default',
                 use_gateway: bool = True):
        super().__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)
        
        # 跨注意力（可变形）
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        
        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)
        
        # FFN
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self,
                target: torch.Tensor,
                reference_points: torch.Tensor,
                value: List[torch.Tensor],
                spatial_shapes: List[Tuple[int, int]],
                attn_mask: Optional[torch.Tensor] = None,
                query_pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            target: 查询特征 [B, Q, C]
            reference_points: 参考点 [B, Q, 1, 4] 或 [B, Q, L, 4]
            value: 多尺度特征值
            spatial_shapes: 各层空间尺寸
            attn_mask: 注意力掩码
            query_pos_embed: 查询位置嵌入
        """
        # 自注意力
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)
        
        # 跨注意力
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes
        )
        
        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)
        
        # FFN
        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))
        
        return target


# ============================================================================
# 分支解码器 (Branch Decoder)
# ============================================================================
class BranchDecoder(nn.Module):
    """
    单个分支的完整解码器
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 num_head: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.,
                 activation: str = 'relu',
                 num_levels: int = 2,  # 每个分支关注的尺度数
                 num_points: int = 4,
                 reg_max: int = 32,
                 cross_attn_method: str = 'default',
                 use_gateway: bool = True,
                 eval_idx: int = -1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.reg_max = reg_max
        
        # 解码器层
        self.layers = nn.ModuleList([
            BranchDecoderLayer(
                hidden_dim, num_head, dim_feedforward, dropout, activation,
                num_levels, num_points, cross_attn_method, use_gateway
            ) for _ in range(num_layers)
        ])
        
        # LQE层
        self.lqe_layers = nn.ModuleList([
            LQE(4, 64, 2, reg_max, act=activation) for _ in range(num_layers)
        ])
    
    def value_op(self, memory, memory_mask, spatial_shapes):
        """预处理编码器输出供MSDeformableAttention使用"""
        if memory_mask is not None:
            memory = memory * memory_mask.to(memory.dtype).unsqueeze(-1)
        value = memory.reshape(memory.shape[0], memory.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)
    
    def forward(self,
                target: torch.Tensor,
                ref_points_unact: torch.Tensor,
                memory: torch.Tensor,
                spatial_shapes: List[Tuple[int, int]],
                bbox_head: nn.ModuleList,
                score_head: nn.ModuleList,
                query_pos_head: nn.Module,
                pre_bbox_head: nn.Module,
                integral: nn.Module,
                project: torch.Tensor,
                reg_scale: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> Dict[str, List[torch.Tensor]]:
        """单个分支的前向传播"""
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, memory_mask, spatial_shapes)
        
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        
        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
        
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)
            
            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()
            
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)
            
            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                
                if not self.training:
                    break
            
            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()
        
        return {
            'bboxes': dec_out_bboxes,
            'logits': dec_out_logits,
            'corners': dec_out_pred_corners,
            'refs': dec_out_refs,
            'pre_bboxes': pre_bboxes,
            'pre_logits': pre_scores,
            'hidden_states': output
        }


# ============================================================================
# 三分支检测头 (Triple Branch Detection Head)
# ============================================================================
class TripleBranchHead(nn.Module):
    """
    三分支检测头：单株、集群、综合
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 num_classes: int = 1,
                 reg_max: int = 32,
                 num_layers_individual: int = 6,
                 num_layers_colony: int = 4,
                 num_layers_combined: int = 6,
                 mlp_act: str = 'relu'):
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # 边界框回归头（分支专用）
        self.bbox_head_individual = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3, act=mlp_act)
            for _ in range(num_layers_individual)
        ])
        self.bbox_head_colony = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3, act=mlp_act)
            for _ in range(num_layers_colony)
        ])
        self.bbox_head_combined = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4 * (reg_max + 1), 3, act=mlp_act)
            for _ in range(num_layers_combined)
        ])
        
        # 分类头（每层）
        self.score_head_individual = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers_individual)
        ])
        self.score_head_colony = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers_colony)
        ])
        self.score_head_combined = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers_combined)
        ])
        
        # 预测头（第一层）
        self.pre_bbox_head_individual = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.pre_bbox_head_colony = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.pre_bbox_head_combined = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        
        # 分类头初始化
        for head in [self.score_head_individual, self.score_head_colony, self.score_head_combined]:
            for layer in head:
                init.constant_(layer.bias, bias)
        
        # 边界框头初始化
        for head in [self.bbox_head_individual, self.bbox_head_colony, self.bbox_head_combined]:
            for layer in head:
                if hasattr(layer, 'layers'):
                    init.constant_(layer.layers[-1].weight, 0)
                    init.constant_(layer.layers[-1].bias, 0)
        
        # 预测头初始化
        for head in [self.pre_bbox_head_individual, self.pre_bbox_head_colony, self.pre_bbox_head_combined]:
            init.constant_(head.layers[-1].weight, 0)
            init.constant_(head.layers[-1].bias, 0)


# ============================================================================
# 完整的双分支DEIM Transformer
# ============================================================================
@register()
class DualBranchDEIMTransformer(nn.Module):
    """
    三分支DEIM Transformer：
    基于多尺度特征融合的三分支解码器，用于多粒度目标检测
    
    主要特性：
    1. 多尺度特征融合：
       - 单株分支：融合 level 0 + level 1（高分辨率）
       - 集群分支：融合 level 1 + level 2（低分辨率）
       - 综合分支：融合 level 0 + level 1 + level 2（全尺度）
    2. 尺度感知查询：不同分支使用不同尺度先验
    3. 独立的解码器和检测头
    """
    
    __share__ = ['num_classes', 'eval_spatial_size']
    
    def __init__(self,
                 num_classes: int = 80,
                 hidden_dim: int = 256,
                 num_queries_individual: int = 300,
                 num_queries_colony: int = 300,
                 num_queries_combined: int = 300,
                 feat_channels: List[int] = [256, 256, 256],
                 feat_strides: List[int] = [8, 16, 32],
                 num_levels: int = 3,
                 num_points: int = 4,
                 nhead: int = 8,
                 num_layers_individual: int = 6,
                 num_layers_colony: int = 4,
                 num_layers_combined: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.,
                 activation: str = 'relu',
                 num_denoising: int = 100,
                 label_noise_ratio: float = 0.5,
                 box_noise_scale: float = 1.0,
                 eval_spatial_size: Optional[Tuple[int, int]] = None,
                 eval_idx_individual: int = -1,
                 eval_idx_colony: int = -1,
                 eval_idx_combined: int = -1,
                 eps: float = 1e-2,
                 aux_loss: bool = True,
                 cross_attn_method: str = 'default',
                 reg_max: int = 32,
                 reg_scale: float = 4.,
                 mlp_act: str = 'relu',
                 use_gateway: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries_individual = num_queries_individual
        self.num_queries_colony = num_queries_colony
        self.num_queries_combined = num_queries_combined
        self.eps = eps
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        
        # 特征投影层
        self._build_input_proj_layer(feat_channels)
        
        # 多尺度特征融合模块
        self.fusion_module = MultiScaleFusionModule(hidden_dim, nhead, dropout)
        
        # 尺度感知查询生成器
        self.query_generator = ScaleAwareQueryGenerator(
            hidden_dim,
            num_queries_individual,
            num_queries_colony,
            num_queries_combined
        )
        
        # 单株分支解码器（关注2个高分辨率尺度）
        self.decoder_individual = BranchDecoder(
            hidden_dim, nhead, num_layers_individual, dim_feedforward,
            dropout, activation, num_levels=2, num_points=num_points,
            reg_max=reg_max, cross_attn_method=cross_attn_method,
            use_gateway=use_gateway, eval_idx=eval_idx_individual
        )
        
        # 集群分支解码器（关注2个低分辨率尺度）
        self.decoder_colony = BranchDecoder(
            hidden_dim, nhead, num_layers_colony, dim_feedforward,
            dropout, activation, num_levels=2, num_points=num_points,
            reg_max=reg_max, cross_attn_method=cross_attn_method,
            use_gateway=use_gateway, eval_idx=eval_idx_colony
        )
        
        # 综合分支解码器（关注全部3个尺度）
        self.decoder_combined = BranchDecoder(
            hidden_dim, nhead, num_layers_combined, dim_feedforward,
            dropout, activation, num_levels=3, num_points=num_points,
            reg_max=reg_max, cross_attn_method=cross_attn_method,
            use_gateway=use_gateway, eval_idx=eval_idx_combined
        )
        
        # 三分支检测头
        self.head = TripleBranchHead(
            hidden_dim, num_classes, reg_max,
            num_layers_individual, num_layers_colony, num_layers_combined, mlp_act
        )
        
        # 其他组件
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        self.integral = Integral(reg_max)
        
        # 查询位置编码头（共享）
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=mlp_act)
        
        # 去噪训练
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])
        
        # 评估层索引
        self.eval_idx_individual = eval_idx_individual if eval_idx_individual >= 0 else num_layers_individual + eval_idx_individual
        self.eval_idx_colony = eval_idx_colony if eval_idx_colony >= 0 else num_layers_colony + eval_idx_colony
        self.eval_idx_combined = eval_idx_combined if eval_idx_combined >= 0 else num_layers_combined + eval_idx_combined
        
        self._reset_parameters(feat_channels)
    
    def _reset_parameters(self, feat_channels):
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                if hasattr(m, '__getitem__') and hasattr(m[0], 'weight'):
                    init.xavier_uniform_(m[0].weight)
    
    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))
                    ]))
                )
    
    def _get_encoder_input(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], Dict]:
        """获取编码器输入并进行多尺度融合"""
        # 投影特征
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # 多尺度融合
        fusion_outputs = self.fusion_module(proj_feats)
        
        return proj_feats, fusion_outputs
    
    def forward(self, feats: List[torch.Tensor], targets=None, is_teacher: bool = False):
        """
        前向传播
        
        Args:
            feats: 骨干网络输出的多尺度特征
            targets: 训练目标
            is_teacher: 是否为教师模型
        
        Returns:
            outputs: 包含两个分支预测结果的字典
        """
        batch_size = feats[0].shape[0]
        device = feats[0].device
        
        # 获取编码器输入和融合特征
        proj_feats, fusion_outputs = self._get_encoder_input(feats)
        
        # 生成尺度感知查询
        queries = self.query_generator(batch_size, device)
        
        attn_mask_individual = None
        attn_mask_colony = None
        attn_mask_combined = None
        dn_meta_individual = None
        dn_meta_colony = None
        dn_meta_combined = None
        
        if self.training and self.num_denoising > 0 and targets is not None:
            # 只在 Combined 分支使用去噪训练
            dn_logits_comb, dn_bbox_comb, attn_mask_comb, dn_meta_comb = \
                get_contrastive_denoising_training_group(
                    targets, self.num_classes, self.num_queries_combined,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale
                )
            if dn_logits_comb is not None:
                queries['combined']['content'] = torch.cat([
                    dn_logits_comb, queries['combined']['content']
                ], dim=1)
                queries['combined']['anchor_unact'] = torch.cat([
                    dn_bbox_comb, queries['combined']['anchor_unact']
                ], dim=1)
                attn_mask_combined = attn_mask_comb
                dn_meta_combined = dn_meta_comb
        
        # 计算投影函数
        project = weighting_function(self.reg_max, self.up, self.reg_scale)
        
        # 单株分支解码
        individual_out = self.decoder_individual(
            queries['individual']['content'],
            queries['individual']['anchor_unact'],
            fusion_outputs['individual'],
            fusion_outputs['spatial_shapes_individual'],
            self.head.bbox_head_individual,
            self.head.score_head_individual,
            self.query_pos_head,
            self.head.pre_bbox_head_individual,
            self.integral,
            project,
            self.reg_scale,
            attn_mask_individual,
            None
        )
        
        # 集群分支解码
        colony_out = self.decoder_colony(
            queries['colony']['content'],
            queries['colony']['anchor_unact'],
            fusion_outputs['colony'],
            fusion_outputs['spatial_shapes_colony'],
            self.head.bbox_head_colony,
            self.head.score_head_colony,
            self.query_pos_head,
            self.head.pre_bbox_head_colony,
            self.integral,
            project,
            self.reg_scale,
            attn_mask_colony,
            None
        )
        
        # 综合分支解码
        combined_out = self.decoder_combined(
            queries['combined']['content'],
            queries['combined']['anchor_unact'],
            fusion_outputs['combined'],
            fusion_outputs['spatial_shapes_combined'],
            self.head.bbox_head_combined,
            self.head.score_head_combined,
            self.query_pos_head,
            self.head.pre_bbox_head_combined,
            self.integral,
            project,
            self.reg_scale,
            attn_mask_combined,
            None
        )
        
        # 处理去噪输出分割（三个分支）
        # Individual 分支
        ind_out_logits = torch.stack(individual_out['logits'])
        ind_out_bboxes = torch.stack(individual_out['bboxes'])
        ind_out_corners = torch.stack(individual_out['corners'])
        ind_out_refs = torch.stack(individual_out['refs'])
        ind_pre_logits = individual_out['pre_logits']
        ind_pre_bboxes = individual_out['pre_bboxes']
        
        dn_out_logits_ind = None
        dn_out_bboxes_ind = None
        dn_out_corners_ind = None
        dn_out_refs_ind = None
        dn_pre_logits_ind = None
        dn_pre_bboxes_ind = None
        
        if dn_meta_individual is not None:
            dn_pre_logits_ind, ind_pre_logits = torch.split(ind_pre_logits, dn_meta_individual['dn_num_split'], dim=1)
            dn_pre_bboxes_ind, ind_pre_bboxes = torch.split(ind_pre_bboxes, dn_meta_individual['dn_num_split'], dim=1)
            dn_out_logits_ind, ind_out_logits = torch.split(ind_out_logits, dn_meta_individual['dn_num_split'], dim=2)
            dn_out_bboxes_ind, ind_out_bboxes = torch.split(ind_out_bboxes, dn_meta_individual['dn_num_split'], dim=2)
            dn_out_corners_ind, ind_out_corners = torch.split(ind_out_corners, dn_meta_individual['dn_num_split'], dim=2)
            dn_out_refs_ind, ind_out_refs = torch.split(ind_out_refs, dn_meta_individual['dn_num_split'], dim=2)
        
        # Colony 分支
        col_out_logits = torch.stack(colony_out['logits'])
        col_out_bboxes = torch.stack(colony_out['bboxes'])
        col_out_corners = torch.stack(colony_out['corners'])
        col_out_refs = torch.stack(colony_out['refs'])
        col_pre_logits = colony_out['pre_logits']
        col_pre_bboxes = colony_out['pre_bboxes']
        
        dn_out_logits_col = None
        dn_out_bboxes_col = None
        dn_out_corners_col = None
        dn_out_refs_col = None
        dn_pre_logits_col = None
        dn_pre_bboxes_col = None
        
        if dn_meta_colony is not None:
            dn_pre_logits_col, col_pre_logits = torch.split(col_pre_logits, dn_meta_colony['dn_num_split'], dim=1)
            dn_pre_bboxes_col, col_pre_bboxes = torch.split(col_pre_bboxes, dn_meta_colony['dn_num_split'], dim=1)
            dn_out_logits_col, col_out_logits = torch.split(col_out_logits, dn_meta_colony['dn_num_split'], dim=2)
            dn_out_bboxes_col, col_out_bboxes = torch.split(col_out_bboxes, dn_meta_colony['dn_num_split'], dim=2)
            dn_out_corners_col, col_out_corners = torch.split(col_out_corners, dn_meta_colony['dn_num_split'], dim=2)
            dn_out_refs_col, col_out_refs = torch.split(col_out_refs, dn_meta_colony['dn_num_split'], dim=2)
        
        # Combined 分支
        comb_out_logits = torch.stack(combined_out['logits'])
        comb_out_bboxes = torch.stack(combined_out['bboxes'])
        comb_out_corners = torch.stack(combined_out['corners'])
        comb_out_refs = torch.stack(combined_out['refs'])
        comb_pre_logits = combined_out['pre_logits']
        comb_pre_bboxes = combined_out['pre_bboxes']
        
        dn_out_logits_comb = None
        dn_out_bboxes_comb = None
        dn_out_corners_comb = None
        dn_out_refs_comb = None
        dn_pre_logits_comb = None
        dn_pre_bboxes_comb = None
        
        if dn_meta_combined is not None:
            dn_pre_logits_comb, comb_pre_logits = torch.split(comb_pre_logits, dn_meta_combined['dn_num_split'], dim=1)
            dn_pre_bboxes_comb, comb_pre_bboxes = torch.split(comb_pre_bboxes, dn_meta_combined['dn_num_split'], dim=1)
            dn_out_logits_comb, comb_out_logits = torch.split(comb_out_logits, dn_meta_combined['dn_num_split'], dim=2)
            dn_out_bboxes_comb, comb_out_bboxes = torch.split(comb_out_bboxes, dn_meta_combined['dn_num_split'], dim=2)
            dn_out_corners_comb, comb_out_corners = torch.split(comb_out_corners, dn_meta_combined['dn_num_split'], dim=2)
            dn_out_refs_comb, comb_out_refs = torch.split(comb_out_refs, dn_meta_combined['dn_num_split'], dim=2)
        
        # 合并三个分支的输出（训练和推理都使用）
        merged_out = self._merge_branch_outputs(
            ind_out_logits[-1], ind_out_bboxes[-1],
            col_out_logits[-1], col_out_bboxes[-1],
            comb_out_logits[-1], comb_out_bboxes[-1],
            batch_size, device,
            iou_threshold=0.5,
            score_threshold=0.01
        )
        
        # 组织输出
        if self.training or is_teacher:
            out = {
                # 单株分支输出
                'pred_logits_individual': ind_out_logits[-1],
                'pred_boxes_individual': ind_out_bboxes[-1],
                'pred_corners_individual': ind_out_corners[-1],
                'ref_points_individual': ind_out_refs[-1],
                
                # 集群分支输出
                'pred_logits_colony': col_out_logits[-1],
                'pred_boxes_colony': col_out_bboxes[-1],
                'pred_corners_colony': col_out_corners[-1],
                'ref_points_colony': col_out_refs[-1],
                
                # 综合分支输出
                'pred_logits_combined': comb_out_logits[-1],
                'pred_boxes_combined': comb_out_bboxes[-1],
                'pred_corners_combined': comb_out_corners[-1],
                'ref_points_combined': comb_out_refs[-1],
                
                # 合并输出（三个分支合并后的结果）
                'pred_logits': merged_out['pred_logits'],
                'pred_boxes': merged_out['pred_boxes'],
                'pred_types': merged_out['pred_types'],
                
                # 解码器隐藏状态
                'dec_hs_individual': individual_out['hidden_states'],
                'dec_hs_colony': colony_out['hidden_states'],
                'dec_hs_combined': combined_out['hidden_states'],
                
                'up': self.up,
                'reg_scale': self.reg_scale
            }
            
            # 辅助输出
            if self.aux_loss:
                out['aux_outputs_individual'] = self._set_aux_loss2(
                    ind_out_logits[:-1], ind_out_bboxes[:-1],
                    ind_out_corners[:-1], ind_out_refs[:-1],
                    ind_out_corners[-1], ind_out_logits[-1]
                )
                out['aux_outputs_colony'] = self._set_aux_loss2(
                    col_out_logits[:-1], col_out_bboxes[:-1],
                    col_out_corners[:-1], col_out_refs[:-1],
                    col_out_corners[-1], col_out_logits[-1]
                )
                out['aux_outputs_combined'] = self._set_aux_loss2(
                    comb_out_logits[:-1], comb_out_bboxes[:-1],
                    comb_out_corners[:-1], comb_out_refs[:-1],
                    comb_out_corners[-1], comb_out_logits[-1]
                )
                out['pre_outputs_individual'] = {
                    'pred_logits': ind_pre_logits,
                    'pred_boxes': ind_pre_bboxes
                }
                out['pre_outputs_colony'] = {
                    'pred_logits': col_pre_logits,
                    'pred_boxes': col_pre_bboxes
                }
                out['pre_outputs_combined'] = {
                    'pred_logits': comb_pre_logits,
                    'pred_boxes': comb_pre_bboxes
                }
            
            # 去噪输出（三个分支都有）
            if dn_meta_individual is not None:
                out['dn_outputs_individual'] = self._set_aux_loss2(
                    dn_out_logits_ind, dn_out_bboxes_ind,
                    dn_out_corners_ind, dn_out_refs_ind,
                    dn_out_corners_ind[-1], dn_out_logits_ind[-1]
                )
                out['dn_pre_outputs_individual'] = {
                    'pred_logits': dn_pre_logits_ind,
                    'pred_boxes': dn_pre_bboxes_ind
                }
                out['dn_meta_individual'] = dn_meta_individual
            
            if dn_meta_colony is not None:
                out['dn_outputs_colony'] = self._set_aux_loss2(
                    dn_out_logits_col, dn_out_bboxes_col,
                    dn_out_corners_col, dn_out_refs_col,
                    dn_out_corners_col[-1], dn_out_logits_col[-1]
                )
                out['dn_pre_outputs_colony'] = {
                    'pred_logits': dn_pre_logits_col,
                    'pred_boxes': dn_pre_bboxes_col
                }
                out['dn_meta_colony'] = dn_meta_colony
            
            if dn_meta_combined is not None:
                out['dn_outputs_combined'] = self._set_aux_loss2(
                    dn_out_logits_comb, dn_out_bboxes_comb,
                    dn_out_corners_comb, dn_out_refs_comb,
                    dn_out_corners_comb[-1], dn_out_logits_comb[-1]
                )
                out['dn_pre_outputs_combined'] = {
                    'pred_logits': dn_pre_logits_comb,
                    'pred_boxes': dn_pre_bboxes_comb
                }
                out['dn_meta_combined'] = dn_meta_combined
        else:
            # 推理时只输出合并后的结果
            out = {
                'pred_logits': merged_out['pred_logits'],
                'pred_boxes': merged_out['pred_boxes'],
                'pred_types': merged_out['pred_types']
            }
        
        return out
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """简单辅助损失（仅分类和边界框）"""
        return [{'pred_logits': a, 'pred_boxes': b} 
                for a, b in zip(outputs_class, outputs_coord)]
    
    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                       teacher_corners=None, teacher_logits=None):
        """完整辅助损失（包含 corners、ref_points 和 teacher 信息）"""
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                 'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]
    
    def _merge_branch_outputs(self, 
                              ind_logits: torch.Tensor, 
                              ind_boxes: torch.Tensor,
                              col_logits: torch.Tensor, 
                              col_boxes: torch.Tensor,
                              comb_logits: torch.Tensor, 
                              comb_boxes: torch.Tensor,
                              batch_size: int, 
                              device: torch.device,
                              iou_threshold: float = 0.5,
                              score_threshold: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        合并三个分支的输出，处理重叠检测。
        
        策略：
        1. 拼接三个分支的所有预测
        2. 计算跨分支的 IoU
        3. 对于高 IoU 的重叠框，保留得分更高的
        4. 输出去重后的结果
        
        Args:
            ind_logits: 单株分支 logits [B, N_ind, num_classes]
            ind_boxes: 单株分支 boxes [B, N_ind, 4]
            col_logits: 集群分支 logits [B, N_col, num_classes]
            col_boxes: 集群分支 boxes [B, N_col, 4]
            comb_logits: 综合分支 logits [B, N_comb, num_classes]
            comb_boxes: 综合分支 boxes [B, N_comb, 4]
            iou_threshold: IoU 阈值，超过此值视为重叠
            score_threshold: 得分阈值，低于此值的预测被忽略
        
        Returns:
            Dict with merged 'pred_logits', 'pred_boxes', 'pred_types'
        """
        # 获取各分支的查询数
        N_ind = ind_logits.shape[1]
        N_col = col_logits.shape[1]
        N_comb = comb_logits.shape[1]
        
        # 拼接所有分支的预测
        all_logits = torch.cat([ind_logits, col_logits, comb_logits], dim=1)  # [B, N_total, C]
        all_boxes = torch.cat([ind_boxes, col_boxes, comb_boxes], dim=1)      # [B, N_total, 4]
        
        # 创建分支类型标记 (0=individual, 1=colony, 2=combined)
        all_types = torch.cat([
            torch.zeros(batch_size, N_ind, device=device),
            torch.ones(batch_size, N_col, device=device),
            torch.full((batch_size, N_comb), 2, device=device)
        ], dim=1)  # [B, N_total]
        
        # 获取得分（取 sigmoid 后的最大类别得分）
        all_scores = all_logits.sigmoid().max(dim=-1)[0]  # [B, N_total]
        
        # 批量处理每个样本
        merged_logits_list = []
        merged_boxes_list = []
        merged_types_list = []
        
        for b in range(batch_size):
            logits_b = all_logits[b]      # [N_total, C]
            boxes_b = all_boxes[b]        # [N_total, 4]
            scores_b = all_scores[b]      # [N_total]
            types_b = all_types[b]        # [N_total]
            
            # 应用得分阈值过滤
            valid_mask = scores_b > score_threshold
            
            if valid_mask.sum() == 0:
                # 没有有效预测，保留所有
                merged_logits_list.append(logits_b)
                merged_boxes_list.append(boxes_b)
                merged_types_list.append(types_b)
                continue
            
            # 计算有效预测之间的 IoU
            valid_boxes = boxes_b[valid_mask]  # [N_valid, 4]
            valid_scores = scores_b[valid_mask]
            valid_indices = torch.where(valid_mask)[0]
            
            # 计算 IoU 矩阵
            iou_matrix = self._compute_box_iou(valid_boxes, valid_boxes)  # [N_valid, N_valid]
            
            # 基于 IoU 和得分进行 NMS 风格的去重
            keep_mask = torch.ones(len(valid_indices), dtype=torch.bool, device=device)
            
            # 按得分降序排列
            sorted_indices = torch.argsort(valid_scores, descending=True)
            
            for i, idx_i in enumerate(sorted_indices):
                if not keep_mask[idx_i]:
                    continue
                    
                for idx_j in sorted_indices[i+1:]:
                    if not keep_mask[idx_j]:
                        continue
                    
                    # 如果 IoU 超过阈值，移除得分较低的
                    if iou_matrix[idx_i, idx_j] > iou_threshold:
                        keep_mask[idx_j] = False
            
            # 构建保留掩码（映射回原始索引）
            final_keep_mask = torch.zeros(len(boxes_b), dtype=torch.bool, device=device)
            final_keep_mask[valid_indices[keep_mask]] = True
            # 也保留得分低于阈值的（让后处理决定）
            final_keep_mask[~valid_mask] = True
            
            # 为了保持固定输出大小，我们不实际过滤，而是降低重复框的得分
            # 这样后处理（如 NMS）可以进一步处理
            suppressed_mask = valid_mask & ~final_keep_mask
            logits_b_merged = logits_b.clone()
            logits_b_merged[suppressed_mask] = -10.0  # 降低被抑制框的得分
            
            merged_logits_list.append(logits_b_merged)
            merged_boxes_list.append(boxes_b)
            merged_types_list.append(types_b)
        
        merged_logits = torch.stack(merged_logits_list, dim=0)
        merged_boxes = torch.stack(merged_boxes_list, dim=0)
        merged_types = torch.stack(merged_types_list, dim=0)
        
        return {
            'pred_logits': merged_logits,
            'pred_boxes': merged_boxes,
            'pred_types': merged_types
        }
    
    def _compute_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        计算两组边界框的 IoU。
        
        Args:
            boxes1: [N, 4] 格式为 (cx, cy, w, h) 归一化坐标
            boxes2: [M, 4] 格式为 (cx, cy, w, h) 归一化坐标
        
        Returns:
            iou: [N, M] IoU 矩阵
        """
        # 转换为 (x1, y1, x2, y2) 格式
        boxes1_xyxy = self._cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = self._cxcywh_to_xyxy(boxes2)
        
        # 计算交集
        lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])  # [N, M, 2]
        rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])  # [N, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        # 计算面积
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])  # [N]
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])  # [M]
        
        # 计算 IoU
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-6)
        
        return iou
    
    def _cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        将 (cx, cy, w, h) 格式转换为 (x1, y1, x2, y2) 格式
        """
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

