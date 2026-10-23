"""
Dual-Branch Hybrid Encoder for Multi-Granularity Object Detection
双分支混合编码器：针对单株和群落目标的差异化特征融合

核心设计（基于RT-DETR/BiFPN/FaPN/GCNet等研究）：
1. 加权双向融合 (BiFPN风格)：可学习的融合权重，替代硬性相加
2. 全局上下文模块 (GCNet/SE风格)：注入全局语义增强小目标可分性
3. 尺度感知门控：让单株/群落权重真正作用到共享特征上
4. [可选] 特征对齐 (可变形卷积)：动态采样校正特征错位
5. [可选] 跨尺度注意力：智能选择/强调不同尺度信息
6. [可选] 分支特化增强：单株用高分辨率增强，群落用大核注意力

分阶段使用建议：
- Stage 1 (基础增强): branch_fusion=True, use_weighted_fusion=True, use_global_context=True
- Stage 2 (加入门控): + use_scale_gate=True, gate_fusion_mode='weighted'
- Stage 3 (特征对齐): + use_feature_align=True
- Stage 4 (分支增强): + use_branch_enhance=True, return_branch_features=True (需配合双头decoder)

Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import copy
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hybrid_encoder import (
    HybridEncoder, 
    ConvNormLayer_fuse, 
    SCDown,
    RepNCSPELAN4, 
    RepNCSPELAN5, 
    CSPLayer,
    VGGBlock,
    TransformerEncoderLayer,
    TransformerEncoder
)
from .utils import get_activation
from ..core import register

__all__ = ['DualBranchHybridEncoder']


# ============================================================================
# Stage 1: 基础增强模块 (推荐先开启)
# ============================================================================

class WeightedFusion(nn.Module):
    """
    加权双向融合 (BiFPN风格)
    
    通过可学习参数调节高层特征和低层特征融合时的占比，
    让网络自行学习最优的多层特征组合。
    
    Reference: EfficientDet (BiFPN)
    """
    
    def __init__(self, 
                 channels: int,
                 num_inputs: int = 2,
                 eps: float = 1e-4,
                 use_conv: bool = False):  # 默认不加额外卷积，减少参数
        super().__init__()
        self.eps = eps
        self.num_inputs = num_inputs
        
        # 可学习融合权重，初始化为1
        self.weights = nn.Parameter(torch.ones(num_inputs))
        
        # 可选的融合后卷积
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True)
            )
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == self.num_inputs
        
        # 确保权重非负并归一化
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        
        # 加权融合
        out = sum(w[i] * inputs[i] for i in range(self.num_inputs))
        
        if self.use_conv:
            out = self.conv(out)
        
        return out


class GlobalContextBlock(nn.Module):
    """
    全局上下文模块 (GCNet/SE风格)
    
    通过全局池化获取整幅图像的语义概况，生成通道权重，
    将全局语义注入局部特征，提升小目标的可分性和检测置信度。
    
    Reference: GCNet, SENet, CBAM
    """
    
    def __init__(self, 
                 channels: int, 
                 reduction: int = 16,
                 use_spatial: bool = False):  # 默认只用通道注意力，更轻量
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        
        # 通道注意力 (SE-like)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力 (可选，默认关闭)
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        ca = self.channel_attention(x)
        out = x * ca
        
        # 空间注意力 (可选)
        if self.use_spatial:
            max_pool = torch.max(out, dim=1, keepdim=True)[0]
            avg_pool = torch.mean(out, dim=1, keepdim=True)
            sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
            out = out * sa
        
        return out + x  # 残差连接


# ============================================================================
# Stage 2: 尺度感知门控 (确认Stage1稳定后开启)
# ============================================================================

class ScaleAwareGate(nn.Module):
    """
    尺度感知门控模块（简化版）
    
    生成单株/群落的软权重，直接作用到共享特征上，
    确保门控参数能获得有效梯度。
    
    gate_fusion_mode:
    - 'weighted': 加权融合 shared = α*gate_ind + (1-α)*gate_col，α可学习
    - 'max': 取最大值 shared = max(gate_ind, gate_col)
    - 'sum': 直接相加 shared = gate_ind + gate_col (需要归一化)
    """
    
    def __init__(self, 
                 channels: int,
                 reduction: int = 8,
                 fusion_mode: str = 'weighted'):
        super().__init__()
        self.fusion_mode = fusion_mode
        mid_channels = max(channels // reduction, 16)
        
        # 共享的特征提取
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # 单株门控头（偏向小目标/高频）
        self.individual_head = nn.Conv2d(mid_channels, channels, 1)
        
        # 群落门控头（偏向大目标/低频）
        self.colony_head = nn.Conv2d(mid_channels, channels, 1)
        
        # 融合权重（weighted模式）
        if fusion_mode == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            gated_feat: 门控后的特征（直接用于后续）
            gate_ind: 单株门控权重（用于可视化/调试）
            gate_col: 群落门控权重（用于可视化/调试）
        """
        # 提取共享特征
        shared = self.shared_conv(x)
        
        # 生成门控权重
        gate_ind = torch.sigmoid(self.individual_head(shared))
        gate_col = torch.sigmoid(self.colony_head(shared))
        
        # 融合门控到原特征
        if self.fusion_mode == 'weighted':
            alpha = torch.sigmoid(self.alpha)  # 确保在[0,1]
            gated_feat = x * (alpha * gate_ind + (1 - alpha) * gate_col)
        elif self.fusion_mode == 'max':
            combined_gate = torch.max(gate_ind, gate_col)
            gated_feat = x * combined_gate
        else:  # 'sum'
            combined_gate = (gate_ind + gate_col) / 2  # 归一化
            gated_feat = x * combined_gate
        
        return gated_feat, gate_ind, gate_col


# ============================================================================
# Stage 3: 特征对齐 (可选，需要验证DCN可用性)
# ============================================================================

class FeatureAlign(nn.Module):
    """
    特征对齐模块 (可变形卷积)
    
    通过学习每个像素邻域的采样偏移，实现几何对齐，
    减少由于尺度变化引起的特征错位。
    
    Reference: FaPN, Deformable ConvNets
    """
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.use_dcn = False
        
        try:
            from torchvision.ops import DeformConv2d
            # offset: 每个卷积核位置2个偏移 (x, y)
            self.offset_conv = nn.Conv2d(
                channels, 2 * kernel_size * kernel_size, 
                kernel_size=kernel_size, padding=padding
            )
            self.deform_conv = DeformConv2d(
                channels, channels, 
                kernel_size=kernel_size, padding=padding
            )
            # 初始化offset为0
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
            self.use_dcn = True
            self.norm = nn.BatchNorm2d(channels)
            self.act = nn.SiLU(inplace=True)
        except (ImportError, Exception) as e:
            print(f"Warning: DeformConv2d not available ({e}), using standard conv")
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_dcn:
            offset = self.offset_conv(x)
            aligned = self.deform_conv(x, offset)
            aligned = self.norm(aligned)
            aligned = self.act(aligned)
        else:
            aligned = self.conv(x)
        
        return aligned + x  # 残差连接


# ============================================================================
# Stage 4: 分支特化增强 (需要配合双头decoder使用)
# ============================================================================

class HighResolutionEnhancement(nn.Module):
    """
    高分辨率增强模块：增强单株分支的细节感知
    
    使用小核卷积和边缘增强保持高频信息
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # 边缘增强（Laplacian-like）
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        with torch.no_grad():
            kernel = torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=torch.float32).view(1, 1, 3, 3)
            self.edge_conv.weight.data = kernel.repeat(channels, 1, 1, 1) * 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.conv(x)
        edge = self.edge_conv(x)
        return x + main + edge * 0.1


class LargeKernelAttention(nn.Module):
    """
    大核注意力模块：增强群落分支的大范围上下文感知
    
    使用深度可分离卷积实现高效的大感受野
    """
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 7,
                 dilation: int = 2):
        super().__init__()
        
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, 
                     dilation=dilation, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


# ============================================================================
# 高级模块 (计算量较大，谨慎开启)
# ============================================================================

class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力模块 (计算量大，建议关闭)
    
    让查询智能地选择或强调某些尺度的信息
    """
    
    def __init__(self,
                 channels: int,
                 num_scales: int = 3,
                 nhead: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        
        self.scale_embed = nn.Parameter(torch.randn(num_scales, channels) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            channels, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        B = feats[0].shape[0]
        
        flat_feats = []
        shapes = []
        for i, feat in enumerate(feats):
            H, W = feat.shape[2:]
            shapes.append((H, W))
            flat = feat.flatten(2).permute(0, 2, 1)
            flat = flat + self.scale_embed[i:i+1]
            flat_feats.append(flat)
        
        all_feats = torch.cat(flat_feats, dim=1)
        attn_out, _ = self.cross_attn(all_feats, all_feats, all_feats)
        attn_out = self.dropout(attn_out)
        all_feats = self.norm(all_feats + attn_out)
        
        out_feats = []
        start = 0
        for i, (H, W) in enumerate(shapes):
            length = H * W
            feat = all_feats[:, start:start+length, :]
            feat = feat.permute(0, 2, 1).reshape(B, self.channels, H, W)
            out_feats.append(feat)
            start += length
        
        return out_feats


# ============================================================================
# 主编码器类
# ============================================================================

@register()
class DualBranchHybridEncoder(HybridEncoder):
    """
    双分支混合编码器：继承自 HybridEncoder，添加多种增强模块
    
    分阶段使用建议：
    - Stage 1: use_weighted_fusion + use_global_context (基础增强)
    - Stage 2: + use_scale_gate (门控融合到shared)
    - Stage 3: + use_feature_align (特征对齐)
    - Stage 4: + use_branch_enhance + return_branch_features (完整双分支)
    
    向下兼容：
    - branch_fusion=False 时，行为与原始 HybridEncoder 完全一致
    """
    
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='deim',
                 csp_type='csp',
                 fuse_op='sum',
                 # ============ 增强模块配置 ============
                 branch_fusion: bool = False,           # 总开关
                 # Stage 1: 基础增强 (推荐先开启)
                 use_weighted_fusion: bool = True,      # BiFPN加权融合
                 use_global_context: bool = True,       # 全局上下文
                 # Stage 2: 门控增强 (确认Stage1稳定后开启)
                 use_scale_gate: bool = False,          # 尺度感知门控
                 gate_fusion_mode: str = 'weighted',    # 门控融合模式: weighted/max/sum
                 # Stage 3: 特征对齐 (可选)
                 use_feature_align: bool = False,       # DCN特征对齐
                 # Stage 4: 分支特化 (需配合双头decoder)
                 use_branch_enhance: bool = False,      # 分支特化增强
                 return_branch_features: bool = False,  # 返回分支特征
                 # 高级选项 (计算量大，谨慎开启)
                 use_cross_scale_attn: bool = False,    # 跨尺度注意力
                 # ============ 模块参数 ============
                 gate_reduction: int = 8,
                 gcb_reduction: int = 16,
                 dcn_kernel_size: int = 3,
                 cross_scale_nhead: int = 4,
                 ):
        # 调用父类初始化
        super().__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_act=enc_act,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            eval_spatial_size=eval_spatial_size,
            version=version,
            csp_type=csp_type,
            fuse_op=fuse_op,
        )
        
        # 保存配置
        self.branch_fusion = branch_fusion
        self.use_weighted_fusion = use_weighted_fusion
        self.use_global_context = use_global_context
        self.use_scale_gate = use_scale_gate
        self.gate_fusion_mode = gate_fusion_mode
        self.use_feature_align = use_feature_align
        self.use_branch_enhance = use_branch_enhance
        self.return_branch_features = return_branch_features
        self.use_cross_scale_attn = use_cross_scale_attn
        
        num_levels = len(in_channels)
        
        if branch_fusion:
            self._build_enhancement_modules(
                hidden_dim, num_levels, nhead, dropout,
                gate_reduction, gcb_reduction, dcn_kernel_size, cross_scale_nhead
            )
            self._print_config()
    
    def _build_enhancement_modules(self, hidden_dim, num_levels, nhead, dropout,
                                   gate_reduction, gcb_reduction, dcn_kernel_size,
                                   cross_scale_nhead):
        """构建增强模块"""
        
        # Stage 1: 加权融合
        if self.use_weighted_fusion:
            self.weighted_fpn = nn.ModuleList([
                WeightedFusion(hidden_dim, num_inputs=2)
                for _ in range(num_levels - 1)
            ])
            self.weighted_pan = nn.ModuleList([
                WeightedFusion(hidden_dim, num_inputs=2)
                for _ in range(num_levels - 1)
            ])
        
        # Stage 1: 全局上下文
        if self.use_global_context:
            self.global_context = nn.ModuleList([
                GlobalContextBlock(hidden_dim, reduction=gcb_reduction)
                for _ in range(num_levels)
            ])
        
        # Stage 2: 尺度感知门控
        if self.use_scale_gate:
            self.scale_gates = nn.ModuleList([
                ScaleAwareGate(hidden_dim, reduction=gate_reduction, 
                              fusion_mode=self.gate_fusion_mode)
                for _ in range(num_levels)
            ])
        
        # Stage 3: 特征对齐
        if self.use_feature_align:
            self.feature_align_fpn = nn.ModuleList([
                FeatureAlign(hidden_dim, kernel_size=dcn_kernel_size)
                for _ in range(num_levels - 1)
            ])
            self.feature_align_pan = nn.ModuleList([
                FeatureAlign(hidden_dim, kernel_size=dcn_kernel_size)
                for _ in range(num_levels - 1)
            ])
        
        # Stage 4: 分支特化增强
        if self.use_branch_enhance:
            self.individual_enhance = nn.ModuleList([
                HighResolutionEnhancement(hidden_dim)
                for _ in range(num_levels)
            ])
            self.colony_enhance = nn.ModuleList([
                LargeKernelAttention(hidden_dim)
                for _ in range(num_levels)
            ])
        
        # 高级: 跨尺度注意力
        if self.use_cross_scale_attn:
            self.cross_scale_attn = CrossScaleAttention(
                hidden_dim, num_scales=num_levels, 
                nhead=cross_scale_nhead, dropout=dropout
            )
    
    def _print_config(self):
        """打印当前配置"""
        print("\n" + "="*60)
        print("DualBranchHybridEncoder 配置")
        print("="*60)
        print(f"Stage 1 - 基础增强:")
        print(f"  use_weighted_fusion: {self.use_weighted_fusion}")
        print(f"  use_global_context: {self.use_global_context}")
        print(f"Stage 2 - 门控增强:")
        print(f"  use_scale_gate: {self.use_scale_gate}")
        if self.use_scale_gate:
            print(f"  gate_fusion_mode: {self.gate_fusion_mode}")
        print(f"Stage 3 - 特征对齐:")
        print(f"  use_feature_align: {self.use_feature_align}")
        print(f"Stage 4 - 分支特化:")
        print(f"  use_branch_enhance: {self.use_branch_enhance}")
        print(f"  return_branch_features: {self.return_branch_features}")
        print(f"高级选项:")
        print(f"  use_cross_scale_attn: {self.use_cross_scale_attn}")
        print("="*60 + "\n")
    
    def forward(self, feats: List[torch.Tensor]) -> Union[List[torch.Tensor], Dict]:
        """
        前向传播
        
        Returns:
            branch_fusion=False: List[Tensor] - 与原始 HybridEncoder 相同
            return_branch_features=False: List[Tensor] - 增强后的共享特征
            return_branch_features=True: Dict - 包含shared/individual/colony/gates
        """
        if not self.branch_fusion:
            return super().forward(feats)
        
        assert len(feats) == len(self.in_channels)
        
        # ============ 1. 输入投影 ============
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # ============ 2. 原始 Transformer 编码器 ============
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                reshaped = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                proj_feats[enc_ind] = reshaped
        
        # ============ 3. 跨尺度注意力（可选，计算量大） ============
        if self.use_cross_scale_attn:
            proj_feats = self.cross_scale_attn(proj_feats)
        
        # ============ 4. 全局上下文 ============
        if self.use_global_context:
            proj_feats = [self.global_context[i](feat) for i, feat in enumerate(proj_feats)]
        
        # ============ 5. 尺度感知门控（作用到proj_feats上） ============
        gates_individual = []
        gates_colony = []
        if self.use_scale_gate:
            for i in range(len(proj_feats)):
                gated_feat, gate_ind, gate_col = self.scale_gates[i](proj_feats[i])
                proj_feats[i] = gated_feat  # 门控直接作用到共享特征
                gates_individual.append(gate_ind)
                gates_colony.append(gate_col)
        
        # ============ 6. FPN 自顶向下融合 ============
        inner_outs = [proj_feats[-1]]
        
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            
            # Lateral conv
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            
            # 上采样
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            
            # 特征对齐（可选）
            fpn_idx = len(self.in_channels) - 1 - idx
            if self.use_feature_align:
                upsample_feat = self.feature_align_fpn[fpn_idx](upsample_feat)
            
            # 融合
            if self.use_weighted_fusion:
                fused = self.weighted_fpn[fpn_idx]([feat_low, upsample_feat])
            else:
                if self.fuse_op == 'sum':
                    fused = upsample_feat + feat_low
                else:
                    fused = torch.concat([upsample_feat, feat_low], dim=1)
            
            # FPN block
            inner_out = self.fpn_blocks[fpn_idx](fused)
            inner_outs.insert(0, inner_out)
        
        # ============ 7. PAN 自底向上融合 ============
        outs = [inner_outs[0]]
        
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            
            # 下采样
            downsample_feat = self.downsample_convs[idx](feat_low)
            
            # 特征对齐（可选）
            if self.use_feature_align:
                downsample_feat = self.feature_align_pan[idx](downsample_feat)
            
            # 融合
            if self.use_weighted_fusion:
                fused = self.weighted_pan[idx]([downsample_feat, feat_high])
            else:
                if self.fuse_op == 'sum':
                    fused = downsample_feat + feat_high
                else:
                    fused = torch.concat([downsample_feat, feat_high], dim=1)
            
            # PAN block
            out = self.pan_blocks[idx](fused)
            outs.append(out)
        
        # ============ 8. 分支特化增强（可选） ============
        if self.use_branch_enhance and self.return_branch_features:
            # 生成分支特化特征
            outs_individual = [self.individual_enhance[i](outs[i].clone()) 
                              for i in range(len(outs))]
            outs_colony = [self.colony_enhance[i](outs[i].clone()) 
                         for i in range(len(outs))]
            
            return {
                'shared': outs,
                'individual': outs_individual,
                'colony': outs_colony,
                'gates': {
                    'individual': gates_individual if gates_individual else None,
                    'colony': gates_colony if gates_colony else None
                }
            }
        
        # ============ 9. 返回结果 ============
        return outs
    
    def get_config_info(self) -> Dict:
        """获取配置信息"""
        info = {
            'branch_fusion': self.branch_fusion,
            'stage1': {
                'use_weighted_fusion': self.use_weighted_fusion,
                'use_global_context': self.use_global_context,
            },
            'stage2': {
                'use_scale_gate': self.use_scale_gate,
                'gate_fusion_mode': self.gate_fusion_mode if self.use_scale_gate else None,
            },
            'stage3': {
                'use_feature_align': self.use_feature_align,
            },
            'stage4': {
                'use_branch_enhance': self.use_branch_enhance,
                'return_branch_features': self.return_branch_features,
            },
            'advanced': {
                'use_cross_scale_attn': self.use_cross_scale_attn,
            }
        }
        
        # 打印BiFPN权重
        if self.branch_fusion and self.use_weighted_fusion:
            fpn_weights = []
            for i, wf in enumerate(self.weighted_fpn):
                w = F.relu(wf.weights).detach().cpu().numpy()
                w_norm = w / (w.sum() + 1e-4)
                fpn_weights.append({
                    'level': i,
                    'weights': w_norm.tolist()
                })
            info['fpn_weights'] = fpn_weights
        
        return info
    
    def get_param_groups(self) -> Dict[str, List]:
        """获取参数分组（用于差异化学习率）"""
        base_params = []
        enhancement_params = []
        
        enhancement_names = ['weighted_', 'global_context', 'scale_gate', 
                            'feature_align', 'individual_enhance', 'colony_enhance',
                            'cross_scale']
        
        for name, param in self.named_parameters():
            if any(x in name for x in enhancement_names):
                enhancement_params.append(param)
            else:
                base_params.append(param)
        
        return {
            'base': base_params,
            'enhancement': enhancement_params
        }
