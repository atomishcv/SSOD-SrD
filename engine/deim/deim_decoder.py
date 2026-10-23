"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import math
import copy
import functools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
from typing import List

from ..core import register
from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid, bias_init_with_prob

from .dfine_decoder import MSDeformableAttention, LQE, Integral
from .dfine_utils import weighting_function, distance2bbox
from .deim_utils import RMSNorm, SwiGLUFFN, Gate, MLP

__all__ = ['DEIMTransformer']


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None,
                 use_gateway=False,
                 ):
        super(TransformerDecoderLayer, self).__init__()

        if layer_scale is not None:
            print(f"     --- Wide Layer@{layer_scale} ---")
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        # ffn
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None):

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes)

        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)

        # ffn
        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu', use_checkpoint=False):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.use_checkpoint = use_checkpoint  # 🔥 Gradient checkpointing 开关
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
                    + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target, # 初始查询内容[B, Q, C]
                ref_points_unact, # 初始查询的未激活边界框[B, Q, 4]
                memory, # 编码器输出特征[B, H*W(i), C] 每个level的特征展平后concat
                spatial_shapes, # 每层特征的空间尺寸 [h, w]
                bbox_head, # 每层的box预测头[L,4*(reg_max+1)]
                score_head, # 每层的score预测头[L,num_classes]
                query_pos_head, # 查询位置编码头[4, hidden_dim]
                pre_bbox_head, # 第0层box预测头[4, hidden_dim]
                integral, # 积分函数
                up, # 上采样因子
                reg_scale, # 回归尺度
                attn_mask=None,
                memory_mask=None,
                dn_meta=None):
        # 初始化query特征；用于残差“冻结”分支与角点预测残差累积
        output = target
        output_detach = pred_corners_undetach = 0
        # 1.预处理encoder输出供MSDeformableAttention使用（按level拆分成多组value）
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        dec_out_hs = []
        # 2.将离散分布(4*(reg_max+1))映射为连续距离的投影函数，用于FDR
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        # 3.参考点反sigmoid激活到(0,1)，再生成查询位置编码
        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)#查询相邻的位置编码
        dec_out_hs = []
        # 4.逐层迭代更新查询内容，生成中间bbox与score
        for i, layer in enumerate(self.layers): #共用5层的transformer
            # cross-attn所需的参考点形状 [B, Q, 1, 4]
            ref_points_input = ref_points_detach.unsqueeze(2)

            # 宽层阶段对通道/空间进行上采样以对齐更大的层（推理保留到eval_idx）
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()
            # transformer 前向传播
            # 🔥 使用 gradient checkpointing 减少显存占用
            if self.training and self.use_checkpoint:
                output = checkpoint(layer, output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed, use_reentrant=False)
            else:
                output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)
            dec_out_hs.append(output)  # 新增：保存当前层隐表示 [B, Q, C]
            
            # 首层：在参考点基础上做一次refine得到初始bbox与score，并冻结为后续迭代的起点
            if i == 0 :
                # 首层：在参考点基础上做一次refine得到初始bbox与score，并冻结为后续迭代的起点
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # FDR细化：预测角点分布并与上一层残差相加，积分->距离，再与初始框组合为中间bbox
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                # 训练：每层都监督；推理：仅在eval_idx层输出
                scores = score_head[i](output)
                # LQE用于结合角点不确定性修正分类质量
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            # 为下一层准备：累计角点残差；把中间bbox作为新的参考点；冻结output分支用于稳定训练
            pred_corners_undetach = pred_corners # 保留当前层角点（带梯度）
            ref_points_detach = inter_ref_bbox.detach() # 冻结中间框作为下一层参考点
            output_detach = output.detach() # 冻结当前层输出用于稳定训练

        dec_hs = {i: h for i, h in enumerate(dec_out_hs)}

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
               torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores, dec_hs


@register()
class DEIMTransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 mlp_act='relu',
                 use_gateway=True,
                 share_bbox_head=False,
                 share_score_head=False,
                 use_checkpoint=False,  # 🔥 Gradient checkpointing 减少显存
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale*hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        # -- print the parameters
        print(f"     --- Use Gateway@{use_gateway} ---")
        print(f"     --- Use Share Bbox Head@{share_bbox_head} ---")
        print(f"     --- Use Share Score Head@{share_score_head} ---")

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, use_gateway=use_gateway)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale, use_gateway=use_gateway)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, num_layers, nhead,
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation,
                                          use_checkpoint=use_checkpoint)  # 🔥 传递 checkpoint 开关
      # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)

        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=mlp_act)#查询相邻的位置编码

        # decoder head
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.integral = Integral(self.reg_max)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        dec_score_head = nn.Linear(hidden_dim, num_classes)
        self.dec_score_head = nn.ModuleList(
            [dec_score_head if share_score_head else copy.deepcopy(dec_score_head) for _ in range(self.eval_idx + 1)]
          + [copy.deepcopy(dec_score_head) for _ in range(num_layers - self.eval_idx - 1)])

        # Share the same bbox head for all layers
        dec_bbox_head = MLP(hidden_dim, hidden_dim, 4 * (self.reg_max+1), 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max+1), 3, act=mlp_act) for _ in range(num_layers - self.eval_idx - 1)])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()


        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, 'layers'):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
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
                        ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # 将各尺度特征通过对应的投影层对齐到 hidden_dim（如 1x1/3x3 卷积 + BN）
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            # 若需要的层数多于输入特征层数，用最后一层逐级下采样/投影补足到 num_levels
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    # 第一补层以原始最后一层为输入
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    # 后续补层以上一补层的输出为输入（递归下采样）
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # 组织编码器输入：展平各层空间维并记录每层空间尺寸
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # 保存每层的空间尺寸 [h, w]，后续生成锚点/位置编码用
            spatial_shapes.append([h, w])

        # 按层在长度维 concat 得到 [b, sum(h*w), c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            
            # 🔥 生成基础 wh（多尺度）
            base_wh = grid_size * (2.0 ** lvl)
            wh = torch.ones_like(grid_xy) * base_wh
            
            # 🔥 根据中心点位置限制 wh，确保锚点框不超出图像边界
            # 约束：cx - w/2 >= 0 且 cx + w/2 <= 1  →  w <= 2 * min(cx, 1-cx)
            # 约束：cy - h/2 >= 0 且 cy + h/2 <= 1  →  h <= 2 * min(cy, 1-cy)
            max_w = 2 * torch.minimum(grid_xy[..., 0], 1 - grid_xy[..., 0])
            max_h = 2 * torch.minimum(grid_xy[..., 1], 1 - grid_xy[..., 1])
            max_wh = torch.stack([max_w, max_h], dim=-1)
            
            # 限制 wh 不超过边界约束，同时保持最小值（避免过小）
            wh = torch.minimum(wh, max_wh)
            wh = torch.maximum(wh, torch.tensor(self.eps, dtype=dtype, device=grid_xy.device))
            
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        
        # 🔥 现在所有锚点都应该在有效范围内，但仍保留检查以确保安全
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        
        # 🔥 对于极少数仍可能无效的锚点（数值误差），使用有限值替代
        max_safe_value = 10.0  # sigmoid(10) ≈ 0.9999
        anchors = torch.where(valid_mask, anchors, max_safe_value)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None,
                           is_teacher: bool = False):

        # 准备解码器的输入（从编码器输出 memory 中选择查询）
        if self.training or self.eval_spatial_size is None:
            # 训练阶段或未缓存 eval 尺寸：按当前特征形状生成锚点与有效掩码
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            # 推理阶段且已缓存：复用预生成的锚点与有效掩码
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            # 多 batch 时，为每个样本复制一份锚点
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # 只保留有效位置的特征，屏蔽掉 padding/无效区域
        memory = valid_mask.to(memory.dtype) * memory

        # 编码器头：从编码器输出上预测“目标性/分数”用于选择查询
        enc_outputs_logits :torch.Tensor = self.enc_score_head(memory)

        # 从编码器特征中按分数选择 Top-K 作为初始查询（以及对应锚点）'默认' 'one2many' '类别无关'
        enc_topk_memory, enc_topk_logits, enc_topk_anchors = \
            self._select_topk(memory, enc_outputs_logits, anchors, self.num_queries)

        # 编码器回归头：为 Top-K 查询预测未激活的 bbox（与锚点相加形成偏移）
        enc_topk_bbox_unact :torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training or is_teacher:
            # 训练时保留 sigmoid 后的 bbox（归一化到 [0,1]），用于辅助监督/可视化
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # 查询内容：可选学习式 query；否则用从编码器选出的特征作为初始查询内容
        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        # 查询的未激活 bbox 与后续噪声（DN）拼接前先截断梯度
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            # 若使用 DN（去噪先验），将其与编码器选取的查询在序列维拼接
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        # 返回：解码器查询内容、查询的初始边界框、编码器的边界框预测、编码器的分类预测
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        """
        原始的全局 Top-K 查询选择策略（恢复版本）
        """
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None

        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors

    def forward(self, feats, targets=None, is_teacher: bool = False,
                denoising_logits=None, denoising_bbox_unact=None, attn_mask=None, dn_meta=None,
                cross_query_label=None, cross_query_bbox=None, cross_attn_mask=None, cross_meta=None):
        """
        Args:
            feats: 编码器输出特征
            targets: 目标标注（用于生成 DN 噪声）
            is_teacher: 是否为教师模型
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta: DN 相关参数（可选）
            cross_query_label: Cross-Query 内容嵌入 [B, cross_pad_size, C]（用于一致性损失）
            cross_query_bbox: Cross-Query 位置嵌入 [B, cross_pad_size, 4]（inverse_sigmoid 空间）
            cross_attn_mask: Cross-Query 注意力掩码
            cross_meta: Cross-Query 元信息字典
        """
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)

        # prepare denoising training (only when targets are provided)构建DN噪声
        if self.num_denoising > 0 and targets is not None and denoising_logits is None and denoising_bbox_unact is None and attn_mask is None and dn_meta is None:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                    )
        elif denoising_logits is not None and denoising_bbox_unact is not None and attn_mask is not None and dn_meta is not None:
            pass  # skip running get_contrastive_denoising_training_group, use the provided ones

        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
        
        # 获取解码器输入（Standard Query + DN Query）
        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact, is_teacher=is_teacher)

        # 🔥 Mix-Query 拼接：[Cross-Query | DN-Query | Standard-Query]
        # Cross-Query 放在最前面，用于一致性损失计算
        mix_meta = None
        if cross_query_label is not None and cross_query_bbox is not None:
            cross_pad_size = cross_query_label.shape[1]
            
            # 拼接内容嵌入：[Cross-Query | DN+Standard-Query]
            init_ref_contents = torch.cat([cross_query_label, init_ref_contents], dim=1)
            # 拼接位置嵌入
            init_ref_points_unact = torch.cat([cross_query_bbox, init_ref_points_unact], dim=1)
            
            # 🔥 构造 Mix-Query 注意力掩码
            # 总长度 = cross_pad_size + (dn_num + num_queries)
            total_size = init_ref_contents.shape[1]
            dn_std_size = total_size - cross_pad_size  # DN + Standard 部分长度
            
            # 初始化新的注意力掩码
            mix_attn_mask = torch.zeros(total_size, total_size, dtype=torch.bool, device=memory.device)
            
            # 1. Cross-Query 部分的掩码（从 cross_attn_mask 复制）
            if cross_attn_mask is not None:
                mix_attn_mask[:cross_pad_size, :cross_pad_size] = cross_attn_mask
            
            # 2. DN+Standard 部分的掩码（从原 attn_mask 复制）
            if attn_mask is not None:
                mix_attn_mask[cross_pad_size:, cross_pad_size:] = attn_mask
            
            # 3. Cross-Query 不能看到 DN+Standard Query（单向屏蔽）
            # Cross-Query 只能看到自己的组内成员
            mix_attn_mask[:cross_pad_size, cross_pad_size:] = True
            
            # 4. DN+Standard Query 不能看到 Cross-Query
            mix_attn_mask[cross_pad_size:, :cross_pad_size] = True
            
            attn_mask = mix_attn_mask
            
            # 🔥 构造 Mix-Query 元信息
            mix_meta = {
                'cross_pad_size': cross_pad_size,  # Cross-Query 长度
                'cross_meta': cross_meta,          # Cross-Query 原始元信息
                'dn_meta': dn_meta,                # DN 元信息
            }
            
            # 更新 dn_meta 的分割点（如果存在）
            if dn_meta is not None:
                # 原始分割: [dn_num, num_queries]
                # 新分割: [cross_pad_size, dn_num, num_queries]
                original_split = dn_meta.get('dn_num_split', [])
                # 🔥 确保 original_split 是列表
                if isinstance(original_split, (list, tuple)):
                    dn_meta['dn_num_split'] = [cross_pad_size] + list(original_split)
                else:
                    # 如果是整数，转换为列表
                    dn_meta['dn_num_split'] = [cross_pad_size, int(original_split)]
                dn_meta['has_cross_query'] = True
            else:
                # 如果没有 DN，创建新的 dn_meta
                dn_meta = {
                    'dn_num_split': [cross_pad_size, self.num_queries],
                    'has_cross_query': True,
                }

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits, dec_hs = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta)

        # 🔥 分离 Cross-Query、DN-Query 和 Standard-Query 的输出
        cross_out_logits = cross_out_bboxes = cross_out_corners = cross_out_refs = None
        cross_pre_logits = cross_pre_bboxes = None
        cross_dec_hs = None
        dn_out_logits = dn_out_bboxes = dn_out_corners = dn_out_refs = None
        dn_pre_logits = dn_pre_bboxes = None
        
        if dn_meta is not None:
            has_cross_query = dn_meta.get('has_cross_query', False)
            split_sizes = dn_meta.get('dn_num_split', [])
            
            # 🔥 确保 split_sizes 是列表
            if not isinstance(split_sizes, (list, tuple)):
                split_sizes = [int(split_sizes)] if isinstance(split_sizes, (int, float)) else []
            
            if has_cross_query and len(split_sizes) == 3:
                # 🔥 三段分割：[cross_pad_size, dn_num, num_queries]
                cross_size, dn_size, std_size = split_sizes
                
                # 分割 pre_logits 和 pre_bboxes (dim=1)
                cross_pre_logits, dn_pre_logits, pre_logits = torch.split(pre_logits, split_sizes, dim=1)
                cross_pre_bboxes, dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, split_sizes, dim=1)
                
                # 分割 out_logits 和 out_bboxes (dim=2)
                cross_out_logits, dn_out_logits, out_logits = torch.split(out_logits, split_sizes, dim=2)
                cross_out_bboxes, dn_out_bboxes, out_bboxes = torch.split(out_bboxes, split_sizes, dim=2)
                
                # 分割 out_corners 和 out_refs (dim=2)
                cross_out_corners, dn_out_corners, out_corners = torch.split(out_corners, split_sizes, dim=2)
                cross_out_refs, dn_out_refs, out_refs = torch.split(out_refs, split_sizes, dim=2)
                
                # 分割 dec_hs (decoder hidden states)
                # dec_hs 可能是 list 或 dict of [B, total_queries, C]
                if dec_hs is not None:
                    if isinstance(dec_hs, dict):
                        # 如果是字典，提取值并保持顺序
                        cross_dec_hs = {k: hs[:, :cross_size, :] for k, hs in dec_hs.items()}
                    elif isinstance(dec_hs, (list, tuple)):
                        cross_dec_hs = [hs[:, :cross_size, :] for hs in dec_hs]
                    else:
                        cross_dec_hs = None
                
            elif has_cross_query and len(split_sizes) == 2:
                # 🔥 两段分割（无 DN）：[cross_pad_size, num_queries]
                cross_size, std_size = split_sizes
                
                # 分割 pre_logits 和 pre_bboxes (dim=1)
                cross_pre_logits, pre_logits = torch.split(pre_logits, split_sizes, dim=1)
                cross_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, split_sizes, dim=1)
                
                # 分割 out_logits 和 out_bboxes (dim=2)
                cross_out_logits, out_logits = torch.split(out_logits, split_sizes, dim=2)
                cross_out_bboxes, out_bboxes = torch.split(out_bboxes, split_sizes, dim=2)
                
                # 分割 out_corners 和 out_refs (dim=2)
                cross_out_corners, out_corners = torch.split(out_corners, split_sizes, dim=2)
                cross_out_refs, out_refs = torch.split(out_refs, split_sizes, dim=2)
                
                # 分割 dec_hs
                if dec_hs is not None:
                    if isinstance(dec_hs, dict):
                        # 如果是字典，提取值并保持顺序
                        cross_dec_hs = {k: hs[:, :cross_size, :] for k, hs in dec_hs.items()}
                    elif isinstance(dec_hs, (list, tuple)):
                        cross_dec_hs = [hs[:, :cross_size, :] for hs in dec_hs]
                    else:
                        cross_dec_hs = None
                
            else:
                # 🔥 原始两段分割（无 Cross-Query）：[dn_num, num_queries]
                dn_pre_logits, pre_logits = torch.split(pre_logits, split_sizes, dim=1)
                dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, split_sizes, dim=1)
                
                dn_out_logits, out_logits = torch.split(out_logits, split_sizes, dim=2)
                dn_out_bboxes, out_bboxes = torch.split(out_bboxes, split_sizes, dim=2)
                
                dn_out_corners, out_corners = torch.split(out_corners, split_sizes, dim=2)
                dn_out_refs, out_refs = torch.split(out_refs, split_sizes, dim=2)

        if self.training or is_teacher:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale, 'dec_hs': dec_hs}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if (self.training or is_teacher) and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1],
                                                     out_corners[-1], out_logits[-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            # 🔥 DN 输出
            if dn_out_logits is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs,
                                                        dn_out_corners[-1], dn_out_logits[-1])
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
            
            if dn_meta is not None:
                out['dn_meta'] = dn_meta
            
            # 🔥 Cross-Query 输出（用于一致性损失计算）
            if cross_out_logits is not None:
                out['cross_outputs'] = {
                    'pred_logits': cross_out_logits[-1],  # [B, cross_pad_size, num_classes]
                    'pred_boxes': cross_out_bboxes[-1],   # [B, cross_pad_size, 4]
                    'pred_corners': cross_out_corners[-1] if cross_out_corners is not None else None,
                    'ref_points': cross_out_refs[-1] if cross_out_refs is not None else None,
                    'dec_hs': cross_dec_hs,  # 用于一致性损失
                }
                out['cross_pre_outputs'] = {'pred_logits': cross_pre_logits, 'pred_boxes': cross_pre_bboxes}
                out['mix_meta'] = mix_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]


    @torch.jit.unused
    def _set_aux_loss2(self, outputs_class, outputs_coord, outputs_corners, outputs_ref,
                       teacher_corners=None, teacher_logits=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                     'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)]