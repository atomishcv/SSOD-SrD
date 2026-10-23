"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None, is_teacher: bool = False, hsv=None,
                cross_query_label=None, cross_query_bbox=None, cross_attn_mask=None, cross_meta=None):
        """
        Args:
            x: RGB输入 [B, 3, H, W]
            targets: 目标标注
            is_teacher: 是否为教师模型
            hsv: HSV输入 [B, 3, H, W]，可选
            cross_query_label: Cross-Query 内容嵌入 [B, cross_pad_size, C]，用于一致性损失
            cross_query_bbox: Cross-Query 位置嵌入 [B, cross_pad_size, 4]（inverse_sigmoid 空间）
            cross_attn_mask: Cross-Query 注意力掩码
            cross_meta: Cross-Query 元信息字典
        """
        x = self.backbone(x)   # DINOv3 backbone，输出多尺度特征
        # 将HSV传递给encoder进行双分支特征融合
        enc_out = self.encoder(x, hsv=hsv)
        
        # 将 cross_query 参数传递给 decoder
        outputs = self.decoder(
            enc_out, targets, is_teacher=is_teacher,
            cross_query_label=cross_query_label,
            cross_query_bbox=cross_query_bbox,
            cross_attn_mask=cross_attn_mask,
            cross_meta=cross_meta
        )

        # 将编码器输出一并返回，供上游使用
        if isinstance(outputs, dict):
            outputs['backbone_feate'] = x
            outputs['encoder_feate'] = enc_out
        return outputs

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
