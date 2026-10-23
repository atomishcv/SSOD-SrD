"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from ..core import register

__all__ = ['HybridEncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        # 使用 affine=True, track_running_stats=True (默认),但确保在 eval 模式下不更新
        self.norm = nn.BatchNorm2d(ch_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        # 显式设置 inplace=False,避免 PyTorch 2.x DDP 的 inplace 操作错误
        self.act = nn.Identity() if act is None else get_activation(act, inpace=False)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:  
            # Simplified: direct conv + norm without clone to save memory
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ForegroundEnhanceBlock(nn.Module):
    """
    Class-agnostic foreground enhancement module designed for single-class detection.
    It boosts foreground responses by conditioning on global prototypes.

    Args:
        channels (int): number of input/output channels.
        reduction (int): channel reduction ratio inside the attention predictor.
        activation (str | None): activation name for hidden convs.
    """
    def __init__(self, channels: int, reduction: int = 4, activation: str = "silu") -> None:
        super().__init__()
        mid_channels = max(channels // reduction, 1)

        self.conv1 = nn.Conv2d(channels * 2, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Identity() if activation is None else get_activation(activation, inpace=False)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, C, H, W] feature map.

        Returns:
            Tensor: [B, C, H, W] feature map with foreground responses enhanced.
        """
        B, C, H, W = x.shape
        global_proto = x.mean(dim=(2, 3), keepdim=True)            # [B, C, 1, 1]
        cond = torch.cat([x, global_proto.expand(-1, -1, H, W)], dim=1)

        att = self.conv1(cond)
        att = self.bn1(att)
        att = self.act(att)
        att = torch.sigmoid(self.conv2(att))

        return x * (1.0 + att)


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        # 使用 affine=True, track_running_stats=True (默认),但确保在 eval 模式下不更新
        self.norm = nn.BatchNorm2d(ch_out, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        # 显式设置 inplace=False,避免 PyTorch 2.x DDP 的 inplace 操作错误
        self.act = nn.Identity() if act is None else get_activation(act, inpace=False)

    def forward(self, x):
        # Simplified: direct conv + norm without clone to save memory
        y = self.norm(self.conv(x))
        return self.act(y)


# self.cv1 = Conv(c1, c2, 1, 1)
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        # 显式设置 inplace=False,避免 PyTorch 2.x DDP 的 inplace 操作错误
        self.act = nn.Identity() if act is None else get_activation(act, inpace=False)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)

class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu",
                 csp_type='csp2',
                 ):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        if csp_type == 'csp2':
            CSPLayer = CSPLayer2
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        x1 = self.cv1(x)
        y = list(x1.chunk(2, 1))
        y2 = self.cv2(y[-1])
        y3 = self.cv3(y2)
        y.extend([y2, y3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        x1 = self.cv1(x)
        y = list(x1.split((self.c, self.c), 1))
        y2 = self.cv2(y[-1])
        y3 = self.cv3(y2)
        y.extend([y2, y3])
        return self.cv4(torch.cat(y, 1))


# This layer is equivalent to RepC3 in YOLOs repo
class CSPLayer2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock,
                 ):
        super(CSPLayer2, self).__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels * 2, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x1 = self.conv1(x)
        y0, y1 = x1.chunk(2, 1)
        merged = y0 + self.bottlenecks(y1)
        return self.conv3(merged)

class RepNCSPELAN5(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)

        self.cv2 = nn.Sequential(CSPLayer2(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv3 = nn.Sequential(CSPLayer2(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        x1 = self.cv1(x)
        y = list(x1.chunk(2, 1))
        y2 = self.cv2(y[-1])
        y3 = self.cv3(y2)
        y.extend([y2, y3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        x1 = self.cv1(x)
        y = list(x1.split((self.c, self.c), 1))
        y2 = self.cv2(y[-1])
        y3 = self.cv3(y2)
        y.extend([y2, y3])
        return self.cv4(torch.cat(y, 1))

# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 显式设置 inplace=False,避免 PyTorch 2.x DDP 的 inplace 操作错误
        self.activation = get_activation(activation, inpace=False)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


class HSVAttentionFusion(nn.Module):
    """
    使用注意力机制融合RGB特征和HSV特征
    """
    def __init__(self, rgb_dim, hsv_dim, act='silu'):
        super().__init__()
        # HSV特征投影到RGB维度
        self.hsv_proj = nn.Sequential(
            nn.Conv2d(hsv_dim, rgb_dim, 1, bias=False),
            nn.BatchNorm2d(rgb_dim),
        )
        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.Conv2d(rgb_dim * 2, rgb_dim // 4, 1, bias=False),
            nn.BatchNorm2d(rgb_dim // 4),
            get_activation(act, inpace=False),
            nn.Conv2d(rgb_dim // 4, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )
    
    def forward(self, rgb_feat, hsv_feat):
        """
        Args:
            rgb_feat: [B, C, H, W] RGB特征
            hsv_feat: [B, hsv_dim, H, W] HSV特征
        Returns:
            fused_feat: [B, C, H, W] 融合后的特征
        """
        # 投影HSV到RGB维度
        hsv_proj = self.hsv_proj(hsv_feat)
        
        # 计算注意力权重
        concat = torch.cat([rgb_feat, hsv_proj], dim=1)  # [B, 2C, H, W]
        attn = self.attention(concat)  # [B, 2, H, W]
        
        # 加权融合
        rgb_weight = attn[:, 0:1, :, :]   # [B, 1, H, W]
        hsv_weight = attn[:, 1:2, :, :]   # [B, 1, H, W]
        
        fused = rgb_feat * rgb_weight + hsv_proj * hsv_weight
        return fused


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 csp_type='csp',
                 fuse_op='cat',
                 enable_fg_attention=False,  # 默认关闭前景注意力
                 fg_reduction=4,
                 fg_use_levels=None,
                 # HSV分支配置
                 use_hsv_branch=False,
                 hsv_hidden_dim=64,
                 hsv_fusion_type='add',  # 'add', 'cat', 'attention'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.fuse_op = fuse_op
        self.enable_fg_attention = enable_fg_attention
        self.use_hsv_branch = use_hsv_branch
        self.hsv_fusion_type = hsv_fusion_type

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if in_channel != hidden_dim:
                proj = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                            ('norm', nn.BatchNorm2d(hidden_dim))
                        ]))
            else:
                proj = nn.Identity()
            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        input_dim = hidden_dim if self.fuse_op == 'sum' else hidden_dim * 2   # deim use sum instead of cat

        Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1)
        SCDown_Conv = nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2))

        c1, c2, c3, c4, num_blocks = input_dim, hidden_dim, hidden_dim*2, round(expansion * hidden_dim // 2), round(3 * depth_mult)
        if version == 'dfine':
            Fuse_Block = RepNCSPELAN4(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act, csp_type=csp_type)
        elif version == 'deim':
            Fuse_Block = RepNCSPELAN5(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act)
        else:       # RT-DETR
            Fuse_Block = CSPLayer(in_channels=c1, out_channels=c2, num_blocks=num_blocks, act=act, \
                                    expansion=expansion, bottletype=VGGBlock)
            Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act)
            SCDown_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(copy.deepcopy(Lateral_Conv))
            self.fpn_blocks.append(copy.deepcopy(Fuse_Block))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(copy.deepcopy(SCDown_Conv))
            self.pan_blocks.append(copy.deepcopy(Fuse_Block))

        if self.enable_fg_attention:
            if fg_use_levels is None:
                fg_use_levels = list(range(len(in_channels)))
            self.fg_use_levels = sorted(fg_use_levels)
            self.fg_blocks = nn.ModuleDict({
                str(idx): ForegroundEnhanceBlock(self.hidden_dim, reduction=fg_reduction, activation=act)
                for idx in self.fg_use_levels
            })
        else:
            self.fg_blocks = None
            self.fg_use_levels = []

        # ========== HSV分支 ==========
        if self.use_hsv_branch:
            # 轻量级HSV特征提取网络
            # 输入: [B, 3, H, W] HSV图像
            # 输出: [B, hsv_hidden_dim, H/8, W/8] 特征图
            self.hsv_branch = nn.Sequential(
                # Stage 1: /2
                ConvNormLayer_fuse(3, hsv_hidden_dim // 2, 3, 2, act=act),
                # Stage 2: /4
                ConvNormLayer_fuse(hsv_hidden_dim // 2, hsv_hidden_dim, 3, 2, act=act),
                # Stage 3: /8
                ConvNormLayer_fuse(hsv_hidden_dim, hsv_hidden_dim, 3, 2, act=act),
                # 增强层
                ConvNormLayer_fuse(hsv_hidden_dim, hsv_hidden_dim, 3, 1, act=act),
            )
            
            # 多尺度HSV特征提取（对应feat_strides: 8, 16, 32）
            self.hsv_multi_scale = nn.ModuleList([
                # Level 0: stride 8 (与hsv_branch输出相同)
                nn.Identity(),
                # Level 1: stride 16 (下采样2x)
                ConvNormLayer_fuse(hsv_hidden_dim, hsv_hidden_dim, 3, 2, act=act),
                # Level 2: stride 32 (下采样4x from hsv_branch output)
                nn.Sequential(
                    ConvNormLayer_fuse(hsv_hidden_dim, hsv_hidden_dim, 3, 2, act=act),
                    ConvNormLayer_fuse(hsv_hidden_dim, hsv_hidden_dim, 3, 2, act=act),
                ),
            ])
            
            # HSV与RGB特征融合层
            if hsv_fusion_type == 'cat':
                # 拼接后通过1x1卷积降维
                self.hsv_fusion = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(hidden_dim + hsv_hidden_dim, hidden_dim, 1, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        get_activation(act, inpace=False)
                    )
                    for _ in range(len(in_channels))
                ])
            elif hsv_fusion_type == 'add':
                # 先将HSV特征投影到hidden_dim，然后相加
                self.hsv_fusion = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(hsv_hidden_dim, hidden_dim, 1, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                    )
                    for _ in range(len(in_channels))
                ])
            elif hsv_fusion_type == 'attention':
                # 使用注意力机制融合
                self.hsv_fusion = nn.ModuleList([
                    HSVAttentionFusion(hidden_dim, hsv_hidden_dim, act=act)
                    for _ in range(len(in_channels))
                ])

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, hsv=None):
        """
        Args:
            feats: RGB backbone输出的多尺度特征列表 [(B,C1,H1,W1), (B,C2,H2,W2), (B,C3,H3,W3)]
            hsv: HSV图像输入 [B, 3, H, W]，可选
        Returns:
            outs: 融合后的多尺度特征列表
        """
        assert len(feats) == len(self.in_channels)
        # Only clone when necessary to avoid OOM
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder 特征增强，只对最后层应用transformer
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                reshaped = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                proj_feats[enc_ind] = reshaped

        # ========== HSV分支特征提取和融合 ==========
        if self.use_hsv_branch and hsv is not None:
            # 提取HSV基础特征 (stride 8)
            hsv_base = self.hsv_branch(hsv)  # [B, hsv_hidden_dim, H/8, W/8]
            
            # 生成多尺度HSV特征
            hsv_multi = []
            for i, scale_module in enumerate(self.hsv_multi_scale):
                if i == 0:
                    hsv_feat = scale_module(hsv_base)  # stride 8
                elif i == 1:
                    hsv_feat = scale_module(hsv_base)  # stride 16
                else:
                    hsv_feat = scale_module(hsv_base)  # stride 32
                hsv_multi.append(hsv_feat)
            
            # 融合HSV特征到RGB特征（此时RGB特征已经经过transformer增强）
            for i in range(len(proj_feats)):
                hsv_feat = hsv_multi[i]
                rgb_feat = proj_feats[i]
                
                # 确保尺寸匹配
                if hsv_feat.shape[2:] != rgb_feat.shape[2:]:
                    hsv_feat = F.interpolate(hsv_feat, size=rgb_feat.shape[2:], 
                                            mode='bilinear', align_corners=False)
                
                # 根据融合类型进行融合
                if self.hsv_fusion_type == 'cat':
                    concat = torch.cat([rgb_feat, hsv_feat], dim=1)
                    proj_feats[i] = self.hsv_fusion[i](concat)
                elif self.hsv_fusion_type == 'add':
                    hsv_proj = self.hsv_fusion[i](hsv_feat)
                    proj_feats[i] = rgb_feat + hsv_proj
                elif self.hsv_fusion_type == 'attention':
                    proj_feats[i] = self.hsv_fusion[i](rgb_feat, hsv_feat)

        if self.enable_fg_attention:
            for lvl in self.fg_use_levels:
                if lvl < len(proj_feats):
                    proj_feats[lvl] = self.fg_blocks[str(lvl)](proj_feats[lvl])

        # broadcasting and fusion (FPN) 自顶向下融合
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            if self.fuse_op == 'sum':
                fused_feat = upsample_feat + feat_low
            else:
                fused_feat = torch.concat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](fused_feat)
            inner_outs.insert(0, inner_out)

        # PAN 自底向上融合
        outs = [inner_outs[0]] 
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if self.fuse_op == 'sum':
                fused_feat = downsample_feat + feat_height
            else:
                fused_feat = torch.concat([downsample_feat, feat_height], dim=1)
            out = self.pan_blocks[idx](fused_feat)
            outs.append(out)

        return outs
