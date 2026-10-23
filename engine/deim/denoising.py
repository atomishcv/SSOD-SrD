"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional

from torchvision.ops import RoIAlign, roi_align

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def _get_model(model):
    """获取原始模型，兼容 DDP 包装"""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model




def get_contrastive_denoising_training_group(targets,  # 目标列表，每个元素包含 'labels' 与 'boxes'
                                             num_classes,  # 类别总数（用于噪声标签采样与填充值）
                                             num_queries,  # 正常匹配查询数量（decoder的匹配查询数）
                                             class_embed,  # 将类别索引嵌入为查询类别向量的可调用模块
                                             num_denoising=100,  # 去噪查询的期望数量（近似值，后续会按组重算）
                                             label_noise_ratio=0.5,  # 标签噪声比例（随机替换一部分标签）
                                             box_noise_scale=1.0,):  # 框噪声尺度（扰动幅度）
    """cnd"""  # 对比去噪训练（contrastive denoising）所需查询与掩码的构造
    if num_denoising <= 0:  # 若不启用去噪，则直接返回空对象
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets]  # 统计每个样本的gt数量
    device = targets[0]['labels'].device  # 取设备信息以创建后续张量

    max_gt_num = max(num_gts)  # 本batch内的最大gt数量
    if max_gt_num == 0:  # 若整个batch没有gt，无法构造去噪目标
        return None, None, None, None

    num_group = num_denoising // max_gt_num  # 组的数量：每组含 2*max_gt_num 个去噪查询（正负各一份）
    num_group = 1 if num_group == 0 else num_group  # 至少为1组
    # pad gt to max_num of a batch
    bs = len(num_gts)  # batch size

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)  # 先用背景/无效类填充
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)  # 先用零框填充（cx,cy,w,h，归一化）
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)  # 真实gt位置的mask

    for i in range(bs):  # 将各样本真实gt拷贝到前num_gt位置
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']  # 类别填充
            input_query_bbox[i, :num_gt] = targets[i]['boxes']  # 框填充（cx,cy,w,h）
            pad_gt_mask[i, :num_gt] = 1  # 有效位置标记为1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])  # 复制为正负两份，并按组扩展
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])  # 对应复制框
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])  # 对应复制mask
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)  # 前半正样，后半负样
    negative_gt_mask[:, max_gt_num:] = 1  # 标记后半部分为负样
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])  # 按组扩展
    positive_gt_mask = 1 - negative_gt_mask  # 正样mask为取反
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask  # 去掉末维并与有效位相与
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]  # 取正样的列索引
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])  # 按样本拆分索引
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)  # 实际的去噪查询总数（正负各max_gt_num，乘组数）

    if label_noise_ratio > 0:  # 施加标签噪声
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)  # 随机挑选部分位置
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)  # 随机新标签
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)  # 只在有效位替换

    if box_noise_scale > 0:  # 施加边框噪声
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)  # 转为 [x1,y1,x2,y2]
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale  # 以宽高的一半为基准的扰动幅度
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0  # 随机正负号
        rand_part = torch.rand_like(input_query_bbox)  # 随机比例
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)  # 负样增强扰动
        known_bbox = known_bbox + (rand_sign * rand_part * diff)  # 应用扰动（避免 inplace）
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)  # 保持在图像归一化范围
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)  # 转回 [cx,cy,w,h]
        # 避免 inplace 操作
        mask_negative = input_query_bbox < 0
        input_query_bbox = torch.where(mask_negative, -input_query_bbox, input_query_bbox)
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)  # 反sigmoid，得到未激活表示

    input_query_logits = class_embed(input_query_class)  # 类别索引映射为查询类别向量（logits或嵌入）

    tgt_size = num_denoising + num_queries  # 总查询数 = 去噪查询 + 匹配查询
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)  # 自注意力掩码
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True  # 匹配查询不可看见去噪查询

    # reconstruct cannot see each other
    for i in range(num_group):  # 同组去噪查询之间互相不可见（防止泄漏）
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True  # 屏蔽右侧其它组
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True  # 屏蔽左侧其它组
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True  # 屏蔽右侧其它组
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True  # 屏蔽左侧其它组

    dn_meta = {  # 记录去噪元信息
        "dn_positive_idx": dn_positive_idx,  # 正样索引（按batch拆分）
        "dn_num_group": num_group,  # 组数量
        "dn_num_split": [num_denoising, num_queries]  # 分割点：前去噪、后匹配
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])  # 调试：类别嵌入形状
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])  # 调试：去噪框形状
    # print(attn_mask.shape) # torch.Size([496, 496])  # 调试：注意力掩码形状

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta  # 返回带噪声的类别查询、带噪声的边界框、注意力掩码与元信息

def roi_rescale(rois, scale_factor):
    cx = (rois[:, 1] + rois[:, 3]) * 0.5
    cy = (rois[:, 2] + rois[:, 4]) * 0.5
    w = rois[:, 3] - rois[:, 1]
    h = rois[:, 4] - rois[:, 2]
    new_w = w * scale_factor
    new_h = h * scale_factor
    x1 = cx - new_w * 0.5
    x2 = cx + new_w * 0.5
    y1 = cy - new_h * 0.5
    y2 = cy + new_h * 0.5
    new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
    return new_rois


class SimpleRoIExtractor(nn.Module):
    """简化版的 SingleRoIExtractor，实现 RoIAlign 特征抽取。"""

    def __init__(self, featmap_strides, out_channels, pool_size=7, sampling_ratio=0, aligned=True, finest_scale=56):
        super().__init__()
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.roi_layers = nn.ModuleList([
            RoIAlign(
                output_size=(pool_size, pool_size),
                spatial_scale=1.0 / stride,
                sampling_ratio=sampling_ratio,
                aligned=aligned
            ) for stride in featmap_strides
        ])

    def map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self, feats, rois, roi_scale_factor=None):
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        device = feats[0].device
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = roi_rescale(rois, roi_scale_factor)

        for lvl in range(num_levels):
            mask = (target_lvls == lvl)
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() == 0:
                roi_feats = roi_feats + 0.0 * feats[lvl].sum()
                continue
            rois_lvl = rois[inds].to(device)
            roi_feats_lvl = self.roi_layers[lvl](feats[lvl], rois_lvl)
            roi_feats[inds] = roi_feats_lvl

        return roi_feats

class MultiLevelRoIAlign(nn.Module):
    """
    将每个 ROI 按尺度映射到对应的金字塔层，在该层上做 RoIAlign，并把结果还原到原始顺序。
    - feats: List[Tensor], 每层 [B, C(=256), H, W]
    - rois:  Tensor[K, 5], [batch_idx, x1, y1, x2, y2]（图像绝对像素坐标）
    返回:
    - roi_feats: Tensor[K, C, out_h, out_w]
    """
    def __init__(self,
                 featmap_strides: List[int],
                 out_size: int = 7,
                 sampling_ratio: int = 2,
                 finest_scale: int = 56,
                 aligned: bool = True):
        super().__init__()
        self.featmap_strides = featmap_strides
        self.spatial_scales = [1.0 / s for s in featmap_strides]
        self.out_size = out_size
        self.sampling_ratio = sampling_ratio
        self.finest_scale = float(finest_scale)
        self.aligned = aligned

    @torch.no_grad()
    def _map_roi_to_level(self, rois: torch.Tensor, num_levels: int) -> torch.Tensor:
        """
        参照 FPN 规则： level = floor( log2( sqrt(w*h) / finest_scale ) )
        并 clamp 到 [0, num_levels-1]
        """
        # rois: [K, 5] -> 取宽高
        widths  = (rois[:, 3] - rois[:, 1]).clamp(min=1e-6)
        heights = (rois[:, 4] - rois[:, 2]).clamp(min=1e-6)
        scales = torch.sqrt(widths * heights)

        target_lvls = torch.floor(torch.log2(scales / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).to(dtype=torch.long)
        return target_lvls

    def forward(self, feats: List[torch.Tensor], rois: torch.Tensor) -> torch.Tensor:
        """
        feats: list of [B, C, H, W]
        rois:  [K, 5]，其中 rois[:, 0] 是 batch 索引
        
        注意: 避免使用 inplace 索引赋值,改用列表收集后 scatter 方式,确保 DDP 兼容性
        """
        assert len(feats) == len(self.featmap_strides), "feats 与 featmap_strides 数量不一致"
        device = feats[0].device
        dtype = feats[0].dtype
        num_levels = len(feats)
        C = feats[0].shape[1]

        if rois.numel() == 0:
            return torch.zeros((0, C, self.out_size, self.out_size), device=device, dtype=dtype)

        # 计算每个 ROI 的目标层
        target_lvls = self._map_roi_to_level(rois, num_levels)

        # 使用列表收集结果,避免 inplace 索引赋值
        K = rois.shape[0]
        pooled_list = []
        idx_list = []

        # 逐层做 roi_align，收集结果和对应的索引
        for lvl in range(num_levels):
            idx_in_level = torch.nonzero(target_lvls == lvl).squeeze(1)
            if idx_in_level.numel() == 0:
                continue
            rois_lvl = rois[idx_in_level]  # [k_l, 5]
            feat_lvl = feats[lvl]          # [B, C, H, W]
            spatial_scale = self.spatial_scales[lvl]

            # 使用函数式 roi_align：input [B,C,H,W], rois [k,5]
            pooled = roi_align(
                input=feat_lvl,
                boxes=rois_lvl,
                output_size=(self.out_size, self.out_size),
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio,
                aligned=self.aligned,
            )  # [k_l, C, out, out]
            
            pooled_list.append(pooled)
            idx_list.append(idx_in_level)

        # 使用列表收集方式组装结果,避免 inplace 索引赋值
        if len(pooled_list) == 0:
            return torch.zeros((K, C, self.out_size, self.out_size), device=device, dtype=dtype)
        
        # 将所有 pooled 结果和索引合并
        all_pooled = torch.cat(pooled_list, dim=0)  # [total_k, C, out, out]
        all_indices = torch.cat(idx_list, dim=0)    # [total_k]
        
        # 创建一个按原始索引排列的列表,避免 inplace 操作
        # 初始化为零张量列表
        out_list = [None] * K
        for i, idx in enumerate(all_indices):
            out_list[idx.item()] = all_pooled[i:i+1]
        
        # 将 None 替换为零张量(处理某些 ROI 可能没有被分配的情况,虽然理论上不会发生)
        for i in range(K):
            if out_list[i] is None:
                out_list[i] = torch.zeros((1, C, self.out_size, self.out_size), device=device, dtype=dtype)
        
        # 拼接为最终输出
        out = torch.cat(out_list, dim=0)

        return out


# =========================
# 2) 轻量 Projector：任意 RoIAlign 尺寸 → hidden_dim 维
# =========================
class Projector(nn.Module):
    """
    将 (N, C=in_channels, H, W) 的 RoI 特征映射到 (N, hidden_dim) 查询向量。
    结构：2×Conv(3x3) → GAP → MLP(in_channels→mid→hidden_dim)，末端不激活。
    """
    def __init__(self, in_channels=192, hidden_dim=192, mid_dim=384, use_layernorm=True, p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(in_channels)

        self.pool = nn.AdaptiveAvgPool2d(1)  # 自适应，支持任意 H×W

        self.fc1 = nn.Linear(in_channels, mid_dim, bias=True)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(mid_dim) if use_layernorm else nn.BatchNorm1d(mid_dim)
        self.drop = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.fc2 = nn.Linear(mid_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_channels, H, W)
        if x.numel() == 0:
            return x.new_zeros((0, self.fc2.out_features))

        # 显式设置 inplace=False,避免 PyTorch 2.x DDP 的 inplace 操作错误
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = F.relu(self.bn2(self.conv2(x)), inplace=False)
        x = self.pool(x).flatten(1)   # (N, in_channels)
        x = self.fc1(x)
        # BatchNorm1d 需要 (N, D)；LayerNorm 任意即可
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)               # (N, hidden_dim)
        return x


# =========================
# 3) 标准标签嵌入（nn.Embedding）
# =========================
class LabelEncoder(nn.Module):
    """
    将类别索引编码成 256 维查询向量；
    支持可选的“背景/填充类”（index= num_classes）。
    """
    def __init__(self, num_classes: int, hidden_dim: int = 192, add_bg_class: bool = True):
        super().__init__()
        vocab_size = num_classes + 1 if add_bg_class else num_classes
        self.add_bg = add_bg_class
        self.num_classes = num_classes
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        labels: LongTensor[...,]，取值范围 [0, num_classes]（若 add_bg=True，则 num_classes 代表背景/填充）
        返回: (..., hidden_dim)
        """
        return self.emb(labels)


# =========================
# 4) 一个小的“拼装器”示例：从多尺度特征和 RoIs 得到 hidden_dim 维查询向量
# =========================
class ConsistencyQueryBuilder(nn.Module):
    """
    把多尺度特征 + RoIs -> (N_roi, 256, out, out) -> projector -> (N_roi, 256)
    用于你之前的：
        consistency_query_embed = self.roi_extractor(mlvl_feats, rois_bboxes)
        consistency_query_embed = self.projector(consistency_query_embed)
        input_query_label_1[(known_bid_1.long(), map_known_indice_1)] = consistency_query_embed
    """
    def __init__(self,
                 featmap_strides: List[int] = [8, 16, 32],
                 out_size: int = 7,
                 sampling_ratio: int = 2,
                 finest_scale: int = 56,
                 aligned: bool = True,
                hidden_dim: int = 192):
        super().__init__()
        self.roi_extractor = MultiLevelRoIAlign(
            featmap_strides=featmap_strides,
            out_size=out_size,
            sampling_ratio=sampling_ratio,
            finest_scale=finest_scale,
            aligned=aligned
        )
        self.projector = Projector(in_channels=hidden_dim, hidden_dim=hidden_dim)

    @staticmethod
    def prepare_feats(feats: List[torch.Tensor], img_metas: Optional[List[dict]] = None):
        """
        与你原代码接口一致的占位函数（如需可在此生成 masks/pos enc）。
        这里直接返回 feats, None, None。
        """
        mlvl_feats = feats
        mlvl_masks = [None] * len(feats)
        mlvl_pos   = [None] * len(feats)
        return mlvl_feats, mlvl_masks, mlvl_pos

    def forward(self,
                feats: List[torch.Tensor],  # [P3, P4, P5, ...]，每层 [B, 256, H, W]
                rois_bboxes: torch.Tensor   # [K, 5]，[batch_idx, x1, y1, x2, y2]
                ) -> torch.Tensor:
        # 1) ROI 对齐
        roi_feats = self.roi_extractor(feats, rois_bboxes)   # [K, 256, 7, 7]
        # 2) 投影为 256-D 查询向量
        query_vec = self.projector(roi_feats)                # [K, 256]
        return query_vec

def _build_sine_positional_encoding(mask: torch.Tensor, num_pos_feats: int = 96, temperature: int = 10000) -> torch.Tensor:
    """
    DETR风格正弦位置编码。输入 mask: [B, H, W] (bool)，输出: [B, 2*num_pos_feats, H, W]
    """
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps)
    x_embed = x_embed / (x_embed[:, :, -1:] + eps)

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=mask.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos

def _prepare_feats_fpn(self, mlvl_feats, img_metas):
    """
    将多尺度特征对齐到 decoder.hidden_dim 通道，生成 mask 与正弦位置编码。
    返回: srcs(List[B, C, H, W]), mlvl_masks(List[B, H, W]), mlvl_pos(List[B, C, H, W])
    
    注意: 该函数应在 torch.no_grad() 上下文中调用,且会使用教师模型的 input_proj
    """
    device = mlvl_feats[0].device
    batch_size = mlvl_feats[0].size(0)
    input_img_h, input_img_w = img_metas[0]['batch_input_shape']
    img_masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w), dtype=torch.float32, device=device)
    for img_id in range(batch_size):
        img_h, img_w = img_metas[img_id]['img_shape']
        img_masks[img_id, :img_h, :img_w] = 0

    # 获取原始模型(兼容 DDP 包装)
    teacher_model = _get_model(self.teacher)
    student_model = _get_model(self.student)
    
    decoder = teacher_model.decoder
    hidden_dim = getattr(student_model.decoder, 'hidden_dim', 192)
    input_proj = decoder.input_proj  # ModuleList
    num_levels = decoder.num_levels

    srcs, mlvl_masks, mlvl_pos = [], [], []
    for l, feat in enumerate(mlvl_feats):
        # 使用教师模型的 input_proj 进行特征投影(在 no_grad 下安全)
        # 确保输出是独立的张量,避免共享内存
        if l < len(input_proj):
            src = input_proj[l](feat).clone()
        else:
            src = feat.clone()
        mask_l = F.interpolate(img_masks[None], size=src.shape[-2:]).to(torch.bool).squeeze(0)
        pos_l = _build_sine_positional_encoding(mask_l, num_pos_feats=hidden_dim // 2)
        srcs.append(src)
        mlvl_masks.append(mask_l)
        mlvl_pos.append(pos_l)

    if num_levels > len(srcs):
        _len_srcs = len(srcs)
        for l in range(_len_srcs, num_levels):
            if l == _len_srcs:
                src = input_proj[l](mlvl_feats[-1]).clone()
            else:
                src = input_proj[l](srcs[-1]).clone()
            srcs.append(src)
            mask_l = F.interpolate(img_masks[None], size=src.shape[-2:]).to(torch.bool).squeeze(0)
            pos_l = _build_sine_positional_encoding(mask_l, num_pos_feats=hidden_dim // 2)
            mlvl_masks.append(mask_l)
            mlvl_pos.append(pos_l)

    return srcs, mlvl_masks, mlvl_pos

def prepare_unsup_cdn(self, teacher_info, student_info, pseudo_bboxes, pseudo_labels, det_bboxes, det_labels, 
                        dn_args=None, hidden_dim=192, num_queries=300, num_classes=80, prior_info=None):
    """同时构造“对比式去噪查询（CDN）”与“无监督一致性去噪查询”。
    Args:
        - pseudo_bboxes: 目标（学生强增广）空间中的伪框。
        - pseudo_labels: 目标空间中的伪标签。
        - det_bboxes: 源（教师弱增广）空间中的原始检测框。
        - det_labels: 源空间中的原始检测标签。
    该函数会同时准备 CDN 与一致性查询，最终 DN 查询的顺序为：
    "[一致性查询, 去噪查询]"
    """
    
    # 目标（学生强增广）与源（教师弱增广）两种视角的元信息
    img_metas_tgt = student_info['meta']
    img_metas_src = teacher_info['meta']

    # 对应两种视角的输入图像，用于后续提取特征/构造查询嵌入
    # imgs_tgt = student_info['image']
    # imgs_src = teacher_info['image']

    norm_pseudo_bboxes_list = []

    # ++++++ consistency part ++++++ #
    for img_meta, pseudo_bbox in zip(img_metas_tgt, pseudo_bboxes):
        img_h, img_w, = img_meta['img_shape']
        
        # 若该图像无伪框，回退为居中大框，避免下游空集合
        if pseudo_bbox.size(0) == 0:
            pseudo_bbox = pseudo_bbox.new_tensor([[img_w / 4, img_h / 4, 3 * img_w/ 4, 3 * img_h / 4]])

        factor = pseudo_bbox.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(
                                        pseudo_bbox.size(0), 1)
        # 归一化伪框到目标空间 [0,1] 的 cx,cy,w,h 表示
        pseudo_bbox_ = box_xyxy_to_cxcywh(pseudo_bbox)
        norm_pseudo_bboxes = pseudo_bbox_ / factor
        norm_pseudo_bboxes = norm_pseudo_bboxes.clamp(min=0.0, max=1.0)
        norm_pseudo_bboxes_list.append(norm_pseudo_bboxes)

    # pseudo label number
    known_num = [bboxes.size(0) for bboxes in norm_pseudo_bboxes_list]

    # 选一个可靠的 device
    device = (norm_pseudo_bboxes_list[0].device
            if len(norm_pseudo_bboxes_list) and norm_pseudo_bboxes_list[0].numel() > 0
            else mlvl_feats[0].device)
    sizes = [t.shape[0] for t in norm_pseudo_bboxes_list]
    if sum(sizes) == 0:
        batch_idx = torch.empty(0, dtype=torch.long, device=device)
    else:
        batch_idx = torch.cat([
            torch.full((n,), i, device=device, dtype=torch.long)
            for i, n in enumerate(sizes)
        ])
    # batch_idx = torch.cat([imgs_tgt.new_full((t.shape[0],), i) for i, t in enumerate(norm_pseudo_bboxes_list)])
    batch_size = len(known_num)

    # 一致性查询的重复组数
    dn_number_1 = 5

    single_pad_1 = int(max(known_num))
    pad_size_1 = int(single_pad_1 * dn_number_1)

    # 一致性查询（位置）构造：不对伪框加噪，直接 inverse_sigmoid 作为 bbox 查询嵌入
    batched_tgt_bboxes = torch.cat(norm_pseudo_bboxes_list)

    known_bid_1 = batch_idx.repeat(dn_number_1, 1).view(-1)
    known_bboxes = batched_tgt_bboxes.repeat(dn_number_1, 1)
    consistency_bbox_embed = inverse_sigmoid(known_bboxes)

    # 预先分配一致性部分的 label/bbox 查询缓冲（按 pad_size_1 进行 padding）
    student_model = _get_model(self.student)
    padding_label = torch.zeros(pad_size_1, getattr(student_model.decoder, 'hidden_dim', hidden_dim)).cuda()
    padding_bbox = torch.zeros(pad_size_1, 4).cuda()

    input_query_label_1 = padding_label.repeat(batch_size, 1, 1)
    input_query_bbox_1 = padding_bbox.repeat(batch_size, 1, 1)

    map_known_indice_1 = torch.tensor([]).to('cuda')
    if len(known_num):
        map_known_indice_1 = torch.cat([torch.tensor(range(num)) for num in known_num])
        map_known_indice_1 = torch.cat([map_known_indice_1 + single_pad_1 * i for i in range(dn_number_1)]).long()

    input_query_bbox_1[(known_bid_1.long(), map_known_indice_1)] = consistency_bbox_embed

    # 一致性查询（内容）构造：以教师弱增广的检测框作为 ROI，抽取并投影成查询特征
    proposal_bboxes = [bbox[:, :4] for bbox in det_bboxes]
    proposal_bboxes_list, loss_weights_list = [], []
    if prior_info is None:
        for img_id, proposal_bbox in enumerate(proposal_bboxes):
            if proposal_bbox.size(0) == 0:
                # 兼容不同键名：'img_metas' 或 'meta'
                meta_list = teacher_info.get('img_metas', teacher_info.get('meta', None))
                if meta_list is None:
                    raise KeyError("teacher_info missing 'img_metas'/'meta'")
                img_shape = meta_list[img_id]['img_shape']
                # 支持 (h,w) 或 (h,w,*) 两种格式
                if isinstance(img_shape, (list, tuple)) and len(img_shape) >= 2:
                    img_h, img_w = img_shape[0], img_shape[1]
                else:
                    img_h, img_w = img_shape
                proposal_bboxes_list.append(proposal_bbox.new_tensor([[img_w / 4, img_h / 4, 3 * img_w / 4, 3 * img_h / 4]]))
                loss_weights_list.append(proposal_bbox.new_zeros(1))
            else:
                proposal_bboxes_list.append(proposal_bbox)
                loss_weights_list.append(proposal_bbox.new_ones(proposal_bbox.size(0)))

        batched_proposal_bboxes = torch.cat(proposal_bboxes_list)
        batched_loss_weights = torch.cat(loss_weights_list)

        loss_weights = batched_loss_weights.unsqueeze(-1).repeat(dn_number_1, 1)

        rois_bboxes = torch.cat([known_bid_1.unsqueeze(-1), batched_proposal_bboxes.repeat(dn_number_1, 1)], dim=-1)

        # 使用 no_grad 处理教师模型的特征提取和投影
        # 这确保了在 DDP 环境下不会有梯度同步问题
        with torch.no_grad():
            mlvl_feats = teacher_info['encoder_feate'] 
            # FPN式对齐（srcs/masks/pos），使通道数与空间尺寸与 decoder 一致
            meta_list = teacher_info.get('img_metas', teacher_info.get('meta', None))
            mlvl_feats, mlvl_masks, mlvl_pos = _prepare_feats_fpn(self, mlvl_feats, meta_list)
            
            # 初始化 roi_extractor (只在第一次调用时创建)
            if not hasattr(self, 'roi_extractor') or self.roi_extractor is None:
                # 从DEIM decoder中获取featmap_strides
                teacher_model = _get_model(self.teacher)
                featmap_strides = getattr(teacher_model.decoder, 'feat_strides', [8, 16, 32, 64])
                self.roi_extractor = MultiLevelRoIAlign(
                    featmap_strides=featmap_strides, out_size=7, sampling_ratio=2, finest_scale=56, aligned=True
                ).to(mlvl_feats[0].device)

            # 构造 rois_bboxes（保持你原来的做法）
            rois_bboxes = torch.cat(
                [known_bid_1.unsqueeze(-1), batched_proposal_bboxes.repeat(dn_number_1, 1)],
                dim=-1
            ).to(mlvl_feats[0].device).float()

            roi_feats = self.roi_extractor(mlvl_feats, rois_bboxes)  # (K, C, 7, 7)
            
            # 统一使用 decoder 的 hidden_dim 作为查询通道数
            student_model = _get_model(self.student)
            dec_hidden_dim = getattr(student_model.decoder, 'hidden_dim', 192)
            
            # 初始化 projector (只在第一次调用或通道数不匹配时创建)
            in_channels = int(roi_feats.shape[1]) if roi_feats.numel() > 0 else dec_hidden_dim
            if (not hasattr(self, 'projector')) or (self.projector is None) or \
               (hasattr(self.projector, 'conv1') and getattr(self.projector.conv1, 'in_channels', None) != in_channels) or \
               (hasattr(self.projector, 'fc2') and getattr(self.projector.fc2, 'out_features', None) != dec_hidden_dim):
                self.projector = Projector(in_channels=in_channels, hidden_dim=dec_hidden_dim).to(roi_feats.device)
            
            # projector 的前向传播也在 no_grad 下进行
            # 这样生成的 consistency_query_embed 不会参与梯度计算
            consistency_query_embed = self.projector(roi_feats).clone()  # (K, dec_hidden_dim)


        input_query_label_1[(known_bid_1.long(), map_known_indice_1)] = consistency_query_embed

    else:
        loss_weights = prior_info['loss_weights']
        input_query_label_1 = prior_info['input_query_label_1']

    # ===== DN（去噪）查询构造 =====
    dn_number_2 = 100

    targets, dn_number_2, label_noise_ratio, box_noise_scale = dn_args

    # 兼容两种格式：
    # - dict: {'boxes': List[Tensor[N,4]], 'labels': List[Tensor[N]]}
    # - list: List[{'boxes': Tensor[N,4], 'labels': Tensor[N]}]
    if isinstance(targets, dict):
        boxes_list = targets['boxes']
        labels_list = targets['labels']
    elif isinstance(targets, list):
        boxes_list = [t['boxes'] for t in targets]
        labels_list = [t['labels'] for t in targets]
    else:
        raise TypeError(f"Unexpected targets type: {type(targets)}")

    num_valid_gts = [t.size(0) for t in boxes_list]

    gt_labels = []
    gt_bboxes = []

    for i, num in enumerate(num_valid_gts):
        if num == 0:
            tmp_bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).cuda()
            tmp_labels = torch.randint(0, 80, (1,)).long().cuda()

            gt_bboxes.append(tmp_bboxes)
            gt_labels.append(tmp_labels)

        else:
            gt_bboxes.append(boxes_list[i])
            gt_labels.append(labels_list[i])

    dn_number_2 = dn_number_2 * 2
    known = [(torch.ones_like(t)).cuda() for t in gt_labels]
    batch_size = len(known)
    known_num = [sum(k) for k in known]
    if int(max(known_num)) == 0:
        dn_number_2 = 1
    else:
        if dn_number_2 >= 100:
            dn_number_2 = dn_number_2 // (int(max(known_num) * 2))
        elif dn_number_2 < 1:
            dn_number_2 = 1
    if dn_number_2 == 0:
        dn_number_2 = 1
    unmask_bbox = unmask_label = torch.cat(known)
    labels = torch.cat([t for t in gt_labels])
    boxes = torch.cat([t for t in gt_bboxes])
    batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(gt_labels)])

    known_indice = torch.nonzero(unmask_label + unmask_bbox)
    known_indice = known_indice.view(-1)

    known_indice = known_indice.repeat(2 * dn_number_2, 1).view(-1)
    known_labels = labels.repeat(2 * dn_number_2, 1).view(-1)
    known_bid_2 = batch_idx.repeat(2 * dn_number_2, 1).view(-1)
    known_bboxs = boxes.repeat(2 * dn_number_2, 1)
    known_labels_expaned = known_labels.clone()
    # 初始化为 known_bboxs，如果有 box_noise_scale，后面会重新赋值
    # 不使用 clone()，直接赋值，避免创建中间张量

    if label_noise_ratio > 0:
        p = torch.rand_like(known_labels_expaned.float())
        chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)
        new_label = torch.randint_like(chosen_indice, 0, num_classes)
        known_labels_expaned.scatter_(0, chosen_indice, new_label)
    single_pad_2 = int(max(known_num))

    pad_size_2 = int(single_pad_2 * 2 * dn_number_2)
    positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number_2, 1)
    positive_idx = positive_idx + (torch.tensor(range(dn_number_2)) * len(boxes) * 2).long().cuda().unsqueeze(1)
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + len(boxes)
    if box_noise_scale > 0:
        # 避免所有 inplace 操作，使用 torch.cat 或 torch.stack
        # known_bbox_ 从 cxcywh 转换为 xyxy
        bbox_xy1 = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
        bbox_xy2 = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
        known_bbox_ = torch.cat([bbox_xy1, bbox_xy2], dim=1)

        # diff 计算（用于噪声缩放）
        diff_xy1 = known_bboxs[:, 2:] / 2
        diff_xy2 = known_bboxs[:, 2:] / 2
        diff = torch.cat([diff_xy1, diff_xy2], dim=1)

        rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part_base = torch.rand_like(known_bboxs)
        # 完全避免 inplace 操作：使用条件选择构建新张量
        # 为 negative samples 添加 1.0 的偏移
        # 使用 isin 或构建 mask 的非 inplace 方式
        if len(negative_idx) > 0:
            # 创建索引张量并检查每个位置是否在 negative_idx 中
            all_indices = torch.arange(len(rand_part_base), device=rand_part_base.device)
            negative_mask = (all_indices.unsqueeze(1) == negative_idx.unsqueeze(0)).any(dim=1)
        else:
            negative_mask = torch.zeros(len(rand_part_base), dtype=torch.bool, device=rand_part_base.device)
        # 使用 where 避免 inplace
        rand_part = torch.where(negative_mask.unsqueeze(1), rand_part_base + 1.0, rand_part_base) * rand_sign
        known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
        known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
        
        # 从 xyxy 转回 cxcywh，避免 inplace 赋值
        bbox_cxy = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
        bbox_wh = known_bbox_[:, 2:] - known_bbox_[:, :2]
        known_bbox_expand = torch.cat([bbox_cxy, bbox_wh], dim=1)
    else:
        # 如果没有 box_noise_scale，直接使用原始 bboxes
        known_bbox_expand = known_bboxs

    m = known_labels_expaned.long().to('cuda')
    # 从DEIM decoder中获取label_enc（使用denoising_class_embed）
    student_model = _get_model(self.student)
    input_label_embed = student_model.decoder.denoising_class_embed(m)
    input_bbox_embed = inverse_sigmoid(known_bbox_expand)

    # 统一使用 decoder 的 hidden_dim
    dec_hidden_dim = getattr(student_model.decoder, 'hidden_dim', 192)
    padding_label = torch.zeros(pad_size_2, dec_hidden_dim).cuda()
    padding_bbox = torch.zeros(pad_size_2, 4).cuda()

    input_query_label_2 = padding_label.repeat(batch_size, 1, 1)
    input_query_bbox_2 = padding_bbox.repeat(batch_size, 1, 1)

    map_known_indice_2 = torch.tensor([]).to('cuda')
    if len(known_num):
        map_known_indice_2 = torch.cat([torch.tensor(range(num)) for num in known_num]) 
        map_known_indice_2 = torch.cat([map_known_indice_2 + single_pad_2 * i for i in range(2 * dn_number_2)]).long()
    if len(known_bid_2):
        input_query_label_2[(known_bid_2.long(), map_known_indice_2)] = input_label_embed
        input_query_bbox_2[(known_bid_2.long(), map_known_indice_2)] = input_bbox_embed

    # 组装输出（与 CND 对齐）
    input_query_logits = torch.cat([input_query_label_1, input_query_label_2], dim=1)
    input_query_bbox_unact = torch.cat([input_query_bbox_1, input_query_bbox_2], dim=1)
    total_dn = input_query_logits.size(1)

    # 注意力掩码：主查询不可见所有 DN（包含一致性+去噪）
    device_mask = input_query_logits.device
    tgt_size = total_dn + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device_mask)
    attn_mask[total_dn:, :total_dn] = True

    # 生成 dn_positive_idx：DN 段内每组前 n_i 为正样（索引基于实际 total_dn）
    dn_positive_idx = []
    for n_i in known_num:
        if int(n_i) == 0:
            dn_positive_idx.append(torch.zeros(0, dtype=torch.long, device=device_mask))
            continue
        per_img = []
        for g in range(dn_number_2):
            start = g * (2 * single_pad_2)
            per_img.append(torch.arange(start, start + int(n_i), device=device_mask))
        dn_positive_idx.append(torch.cat(per_img, dim=0))

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": int(dn_number_2),
        "dn_num_split": [total_dn, num_queries],
        # 供一致性损失对齐使用（学生/教师互检）
        "pad_size_1": int(pad_size_1) if 'pad_size_1' in locals() else int(0),
        "pad_size_2": int(pad_size_2) if 'pad_size_2' in locals() else int(0),
        "num_dn_group_1": int(dn_number_1) if 'dn_number_1' in locals() else int(0),
        "num_dn_group_2": int(dn_number_2) if 'dn_number_2' in locals() else int(0),
        "known_bid_1": known_bid_1 if 'known_bid_1' in locals() else torch.tensor([]).to(device_mask),
        "known_bid_2": known_bid_2 if 'known_bid_2' in locals() else torch.tensor([]).to(device_mask),
        "map_known_indice_1": map_known_indice_1 if 'map_known_indice_1' in locals() else torch.tensor([]).to(device_mask),
        "map_known_indice_2": map_known_indice_2 if 'map_known_indice_2' in locals() else torch.tensor([]).to(device_mask),
        "loss_weights": loss_weights.to(device_mask) if 'loss_weights' in locals() else None,
        "input_query_label_1": input_query_label_1 if 'input_query_label_1' in locals() else None,
    }

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta

