"""
Projector 模块（参考 Semi-DETR）

将 RoI 特征投影到 Transformer Query 空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class Projector(nn.Module):
    """
    RoI 特征投影器（参考 Semi-DETR）
    
    将从 backbone 特征中提取的 RoI 特征投影到 Transformer Query 空间。
    结构：Conv → BN → ReLU → Conv → BN → ReLU → Flatten → FC → BN → ReLU → FC → ReLU
    """
    
    def __init__(self, in_channels=256, hidden_dim=256, roi_size=7):
        super().__init__()
        self.roi_size = roi_size
        
        # Conv 层：提取空间特征
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.ac1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.ac2 = nn.ReLU(inplace=True)
        
        # FC 层：降维到 query 维度
        fc_input_dim = in_channels * roi_size * roi_size  # 256 * 7 * 7 = 12544
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input_dim, 1024)
        self.bn_fc = nn.BatchNorm1d(1024)
        self.fc_relu1 = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(1024, hidden_dim)
        self.fc_relu2 = nn.ReLU(inplace=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, roi_features):
        """
        Args:
            roi_features: RoI 特征 [N, C, roi_size, roi_size]
            
        Returns:
            query_embed: Query 嵌入 [N, hidden_dim]
        """
        # Conv 层
        x = self.ac1(self.bn1(self.conv1(roi_features)))
        x = self.ac2(self.bn2(self.conv2(x)))
        
        # FC 层
        x = self.flatten(x)
        x = self.fc_relu1(self.bn_fc(self.fc1(x)))
        x = self.fc_relu2(self.fc2(x))
        
        return x


class MultiScaleRoIExtractor(nn.Module):
    """
    多尺度 RoI 特征提取器（参考 Semi-DETR）
    
    从多尺度特征图中提取 RoI 特征
    """
    
    def __init__(self, 
                 featmap_strides=[8, 16, 32],
                 out_channels=256,
                 roi_size=7,
                 sampling_ratio=0):
        super().__init__()
        self.featmap_strides = featmap_strides
        self.out_channels = out_channels
        self.roi_size = roi_size
        self.sampling_ratio = sampling_ratio
        
        # 根据特征图 stride 决定使用哪个层级
        self.finest_scale = 56  # 参考 FPN，小于此面积使用最细特征
    
    def forward(self, feats, rois):
        """
        Args:
            feats: 多尺度特征列表 [(B, C, H1, W1), (B, C, H2, W2), ...]
            rois: RoI 坐标 [N, 5]，格式 (batch_idx, x1, y1, x2, y2)，像素坐标
            
        Returns:
            roi_feats: RoI 特征 [N, C, roi_size, roi_size]
        """
        num_levels = len(feats)
        
        if rois.shape[0] == 0:
            return rois.new_zeros(0, self.out_channels, self.roi_size, self.roi_size)
        
        # 根据 RoI 面积分配到不同层级
        roi_feats = rois.new_zeros(rois.shape[0], self.out_channels, self.roi_size, self.roi_size)
        
        # 计算每个 RoI 的面积
        widths = rois[:, 3] - rois[:, 1]
        heights = rois[:, 4] - rois[:, 2]
        areas = widths * heights
        
        # 根据面积决定使用哪个层级
        # 使用 FPN 的公式：k = floor(k0 + log2(sqrt(wh)/224))
        # 简化版本：直接根据面积阈值分配
        target_lvls = torch.floor(torch.log2(torch.sqrt(areas) / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        
        for lvl in range(num_levels):
            inds = (target_lvls == lvl).nonzero(as_tuple=True)[0]
            if inds.numel() == 0:
                continue
            
            rois_lvl = rois[inds]
            spatial_scale = 1.0 / self.featmap_strides[lvl]
            
            roi_feats_lvl = roi_align(
                feats[lvl],
                rois_lvl,
                output_size=self.roi_size,
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=True
            )
            
            roi_feats[inds] = roi_feats_lvl
        
        return roi_feats

