"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer
import os

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) 特征增强模块
    包含通道注意力和空间注意力，用于增强特征表示
    显存友好，计算复杂度低
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # 先应用通道注意力
        out = x * self.channel_attention(x)
        # 再应用空间注意力
        out = out * self.spatial_attention(out)
        return out

class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4


@register()
class DINOv3STAs(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_sta=True,
        conv_inplane=16,
        hidden_dim=None,
    ):
        super(DINOv3STAs, self).__init__()
        if 'dinov3' in name:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            dinov3_path = os.path.join(project_root, 'dinov3')
            print(f'Loading DINOv3 from: {dinov3_path}')
            self.dinov3 = torch.hub.load(dinov3_path, name, source='local', weights=weights_path)
            while len(self.dinov3.blocks) != (interaction_indexes[-1] + 1):
                del self.dinov3.blocks[-1]
            del self.dinov3.head
        else:
            self.dinov3 =  VisionTransformer(embed_dim=embed_dim, num_heads=num_heads, return_layers=interaction_indexes)
            if weights_path is not None:
                print(f'Loading ckpt from {weights_path}...')
                checkpoint = torch.load(weights_path)
                self.dinov3._model.load_state_dict(checkpoint)
            else:
                print('Training ViT-Tiny from scratch!')

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
            self.cbam_fusion = nn.ModuleList([
                CBAM(in_channels=embed_dim + conv_inplane * 2, reduction=8, kernel_size=7),  # 1/8 尺度
                CBAM(in_channels=embed_dim + conv_inplane * 4, reduction=8, kernel_size=7),  # 1/16 尺度
                CBAM(in_channels=embed_dim + conv_inplane * 4, reduction=8, kernel_size=7),  # 1/32 尺度
            ])
            print("Using CBAM for feature fusion (memory efficient): cat first, then CBAM")
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        if use_sta:
            # CBAM 增强后拼接：embed_dim + detail_dim
            self.convs = nn.ModuleList([
                nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            ])
        else:
            # 不使用 STA 时，直接使用语义特征
            self.convs = nn.ModuleList([
                nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])
    def forward(self, x):
        # 计算教师输出特征图的高宽（DINOv3主干默认 stride=16）  
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        # 计算 patch_token 的网格尺寸（patch_size 通常是 14 或 16）  
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        # 展开输入张量的 batch、通道与空间尺寸  
        bs, C, h, w = x.shape

        # 若设置了 interaction_indexes 且 backbone 不是标准 VisionTransformer，则走自定义提层逻辑  
        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            # 提取指定层的中间特征，返回 (特征, cls_token) 列表  
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            # 否则直接执行前向，返回默认特征序列  
            all_layers = self.dinov3(x)

        # 若只得到一层特征，复制三份，保证后续有三尺度输出  
        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]
        
        sem_feats = []
        # 计算尺度数量（期望得到 3 个尺度：大、中、小）  
        num_scales = len(all_layers) - 2
        # 遍历每一层的语义特征并按尺度对齐  
        for i, sem_feat in enumerate(all_layers):
            # 拆出 patch token（忽略 cls_token），形状 [B, N, D]  
            feat, _ = sem_feat
            # 还原为 [B, D, Hc, Wc] 的语义特征图  
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
            # 计算当前尺度的目标大小（倍数递增/递减）  
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            # 通过插值调整到目标尺寸，确保多尺度对齐  
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            # 存入语义特征列表  
            sem_feats.append(sem_feat)

        # 根据配置决定是否与 STA（细节分支）融合  
        fused_feats = []
        if self.use_sta:
            # 细节分支提取的多尺度特征  
            detail_feats = self.sta(x)
            for i, (sem_feat, detail_feat) in enumerate(zip(sem_feats, detail_feats)):
                # 若语义特征与细节特征尺寸不一致，先做对齐  
                if sem_feat.shape[2:] != detail_feat.shape[2:]:
                    detail_feat = F.interpolate(
                        detail_feat, 
                        size=sem_feat.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                # 通道拼接语义与细节特征
                concat_feat = torch.cat([sem_feat, detail_feat], dim=1)
                fused_feat = self.cbam_fusion[i](concat_feat)
                fused_feats.append(fused_feat)
        else:
            # 不使用 STA 时直接沿用语义特征  
            fused_feats = sem_feats

        # 经过 1×1 卷积 + BN 规范化输出尺度 1（通常对应 C2）  
        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        # 输出尺度 2（对应 C3）  
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        # 输出尺度 3（对应 C4）  
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4