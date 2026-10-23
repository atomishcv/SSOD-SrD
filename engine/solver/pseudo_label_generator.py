"""
伪标签生成模块
用于半监督学习中的教师模型推理和伪标签生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

# 仅需定义一次
def _denorm_to_bgr(img_tensor):
    """
    将 [3,H,W]、ImageNet 归一化的张量反归一化为 OpenCV BGR uint8 图像。
    """
    import torch, cv2
    img = img_tensor.detach().float().cpu()
    if img.ndim != 3 or img.size(0) not in (1, 3):
        raise ValueError("Expect CHW image tensor normalized by ImageNet stats.")
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img * std + mean).clamp(0, 1)                 # 回到 [0,1]
    img = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')  # HWC, RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)           # OpenCV 用 BGR
    return img

def generate_pseudo_labels(model: torch.nn.Module,
                          images: torch.Tensor,
                          targets_meta: List[Dict],
                          device: torch.device,
                          score_thresh: float = 0.0,
                          nms_iou: float = 0.5,
                          topk: int = 300,
                          is_teacher: bool = True,
                          as_batched_dict: bool = True,
                          stack_tensors: bool = True,
                          requires_grad: bool = False,
                          adaptive_proto = None,
                          use_prototype_filter: bool = False,
                          proto_percentile: int = 25,
                          conf_thresh: float = 0.3,
                          fixed_threshold: float = None) -> List[Dict]:
    """
    使用模型生成伪标签
    
    Args:
        model: 模型（教师模型或学生模型）
        images: 图像 [B, C, H, W]
        targets_meta: 目标元数据列表，用于获取orig_size
        device: 设备
        score_thresh: 分数阈值（未使用，保留兼容性）
        nms_iou: NMS IoU阈值，去除重叠框（推荐0.5~0.7）
                - 0: 不使用NMS
                - >0: 使用NMS过滤IoU>阈值的重叠框
        topk: 保留的最大检测数量（未使用，保留兼容性）
        is_teacher: 是否为教师模型（True: eval模式, False: train模式）
        requires_grad: 是否需要保留梯度（学生模型用于计算损失时应设为True）
        adaptive_proto: （已弃用，保留兼容性）
        use_prototype_filter: （已弃用，保留兼容性）
        proto_percentile: （已弃用，保留兼容性）
        conf_thresh: 置信度过滤阈值（0.1~0.3，过滤低分检测）
        fixed_threshold: （已弃用，保留兼容性）
        
    Returns:
        当 as_batched_dict=False：返回按图片的列表 list[dict]
        当 as_batched_dict=True：返回按关键字聚合的批字典；
            若 stack_tensors=True，则对可对齐的张量型字段自动堆叠为 [B, ...] 张量，否则保留为长度为 B 的列表
    """
    if is_teacher:
        model.eval()
        # # 学生模型保持训练状态，不调用model.eval()
        # # ========== 🔴 临时测试代码 开始 🔴 ==========
        # # 用于测试特定图片，完成后请删除此部分（从 🔴 开始 到 🔴 结束）
        # import cv2
        # import numpy as np
        # from PIL import Image
        # import torchvision.transforms as T
        
        # test_image_path = "visualizations/teacher_pseudo/tensor([474], device='cuda:0')_pseudo.jpg"
        
        # try:
        #     # 加载图片
        #     pil_img = Image.open(test_image_path).convert('RGB')
            
        #     # 图像预处理（与训练时一致）
        #     transform = T.Compose([
        #         T.Resize((640, 640)),  # 调整大小
        #         T.ToTensor(),          # 转换为 [0,1] 的张量
        #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet归一化
        #     ])
            
        #     img_tensor = transform(pil_img).to(device)  # [C, H, W]
            
        #     # 替换输入图片（保持批次维度）
        #     images = img_tensor.unsqueeze(0).expand(images.size(0), -1, -1, -1)  # [B, C, H, W]
            
        #     print(f"✅ 临时测试：已加载图片 {test_image_path}")
        #     print(f"   图片形状: {images.shape}")
        # except Exception as e:
        #     print(f"⚠️ 临时测试图片加载失败: {e}，使用原始输入")
        # # ========== 🔴 临时测试代码 结束 🔴 ==========
    
    # ========== 关键修复：根据 requires_grad 决定是否使用 no_grad() ==========
    if requires_grad:
        # 学生模型需要保留梯度用于反向传播
        outputs = model(images, is_teacher=is_teacher)
    else:
        # 教师模型或仅用于推理的情况不需要梯度
        with torch.no_grad():
            outputs = model(images, is_teacher=is_teacher)
    
    # 解析输出
    pred_logits = outputs['pred_logits']  # [B, Q, C]
    pred_boxes = outputs['pred_boxes']    # [B, Q, 4] (cxcywh norm)
    B, Q, C = pred_logits.shape
    pseudo_targets = []
    
    for b in range(B):
        logits_b = pred_logits[b]  # [Q, C]
        boxes_cxcywh = pred_boxes[b]  # [Q, 4]

        _, H_curr, W_curr = images[b].shape  # CHW
        scale_curr = torch.tensor([W_curr, H_curr, W_curr, H_curr], device=boxes_cxcywh.device, dtype=boxes_cxcywh.dtype)

        meta = {}
        if targets_meta is not None and isinstance(targets_meta[b], dict):
            meta = targets_meta[b].get('meta', {})

        if is_teacher:
            # 教师路径：与 test_single_image.py 流程保持一致
            probs = logits_b.sigmoid()  # [Q, C]
            scores, labels = probs.max(dim=-1)  # [Q]

            if scores.numel() == 0:
                pseudo = {
                    'image': images[b],
                    'image_id': targets_meta[b]['image_id'],
                    'orig_size': targets_meta[b]['orig_size'],
                    'boxes': boxes_cxcywh[:0].detach(),
                    'labels': logits_b.new_zeros((0,), dtype=torch.long).detach(),
                    'scores': logits_b.new_zeros((0,)).detach(),
                    'meta': meta,
                }
                pseudo_targets.append(pseudo)
                continue

            # ========== 双重过滤策略：置信度过滤 + NMS去重 ==========
            # 第1步：置信度过滤（去除明显的噪声query）
            conf_keep = scores >= conf_thresh
            
            # 如果置信度过滤后一个都不剩，保留最高分的
            if conf_keep.sum() == 0:
                top_idx = torch.argmax(scores)
                conf_keep = torch.zeros_like(scores, dtype=torch.bool)
                conf_keep[top_idx] = True
            
            # 应用置信度过滤
            conf_boxes = boxes_cxcywh[conf_keep]  # [N, 4]
            conf_labels = labels[conf_keep]       # [N]
            conf_scores = scores[conf_keep]       # [N]
            
            # 第2步：NMS过滤（去除重叠框）
            if nms_iou > 0 and len(conf_boxes) > 0:
                from torchvision.ops import batched_nms
                
                # 转换为xyxy格式 (NMS需要)
                boxes_xyxy = torch.cat([
                    conf_boxes[:, :2] - conf_boxes[:, 2:] / 2,  # x1, y1
                    conf_boxes[:, :2] + conf_boxes[:, 2:] / 2   # x2, y2
                ], dim=-1)
                
                # 执行NMS
                nms_keep = batched_nms(
                    boxes=boxes_xyxy,
                    scores=conf_scores,
                    idxs=conf_labels,
                    iou_threshold=nms_iou
                )
                
                filtered_boxes = conf_boxes[nms_keep]
                filtered_labels = conf_labels[nms_keep]
                filtered_scores = conf_scores[nms_keep]
            else:
                # 不使用NMS
                filtered_boxes = conf_boxes
                filtered_labels = conf_labels
                filtered_scores = conf_scores
            
            # ========== 按分数降序排列 ==========
            order = torch.argsort(filtered_scores, descending=True)
            final_boxes = filtered_boxes[order]
            final_labels = filtered_labels[order]
            final_scores = filtered_scores[order]
            
            # 保险：如果最终没有框，保留原始最高分的框（避免多GPU不一致）
            if final_boxes.numel() == 0:
                top_idx = torch.argmax(scores)
                final_boxes = boxes_cxcywh[top_idx:top_idx+1]
                final_labels = labels[top_idx:top_idx+1]
                final_scores = scores[top_idx:top_idx+1]
            
            pseudo = {
                'image': images[b],
                'image_id': targets_meta[b]['image_id'],
                'orig_size': targets_meta[b]['orig_size'],
                'boxes': final_boxes.detach(),
                'labels': final_labels.detach(),
                'scores': final_scores.detach(),
                'meta': meta,
            }
            pseudo_targets.append(pseudo)

        # VIS_TEACHER_DIR = 'visualizations/teacher_pseudo'  # 你也可以改成配置传入
        # import os, cv2, numpy as np
        # from pathlib import Path

        # if is_teacher and VIS_TEACHER_DIR:
        #     # 仅主进程保存，避免多卡重复；如果没有 dist_utils 就直接跳过判断
        #     do_save = True
        #     if do_save:
        #         Path(VIS_TEACHER_DIR).mkdir(parents=True, exist_ok=True)

        #         # 反归一化得到 BGR 图像 (H, W, 3)
        #         vis_img = _denorm_to_bgr(images[b])
        #         H, W = vis_img.shape[:2]  # 一般是 640x640

        #         # 将 final_boxes (cxcywh, norm) 转成像素 xyxy
        #         if final_boxes.numel() > 0:
        #             final_xyxy = torch.cat(
        #                 [final_boxes[:, :2] - final_boxes[:, 2:] / 2,
        #                 final_boxes[:, :2] + final_boxes[:, 2:] / 2],
        #                 dim=-1
        #             )
        #             # 之前已经有 scale_curr = tensor([W_curr, H_curr, W_curr, H_curr], ...)
        #             final_xyxy_pix = (final_xyxy * scale_curr).round().int()
        #             final_xyxy_pix = torch.stack([
        #                 final_xyxy_pix[:, 0].clamp(0, W - 1),
        #                 final_xyxy_pix[:, 1].clamp(0, H - 1),
        #                 final_xyxy_pix[:, 2].clamp(0, W - 1),
        #                 final_xyxy_pix[:, 3].clamp(0, H - 1),
        #             ], dim=-1)

        #             # 画框 + 分数/类别
        #             for (x1, y1, x2, y2), s, cls in zip(
        #                 final_xyxy_pix.cpu().numpy(),
        #                 final_scores.detach().cpu().numpy(),
        #                 final_labels.detach().cpu().numpy()
        #             ):
        #                 color = (0, 255, 0)  # 教师：绿色
        #                 cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        #                 txt = f"{int(cls)}:{float(s):.2f}"
        #                 (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #                 y1_txt = max(0, int(y1) - 3)
        #                 cv2.rectangle(vis_img, (int(x1), y1_txt - th - 2), (int(x1) + tw + 2, y1_txt), color, -1)
        #                 cv2.putText(vis_img, txt, (int(x1) + 1, y1_txt - 2),
        #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        #         # 组织文件名：优先用 meta['file_name']，否则用 image_id
        #         stem = None
        #         if isinstance(meta, dict) and isinstance(meta.get('file_name', None), str):
        #             stem = Path(meta['file_name']).stem
        #         if stem is None:
        #             stem = str(targets_meta[b].get('image_id', f'img{b}'))
        #         out_path = os.path.join(VIS_TEACHER_DIR, f"{stem}_pseudo.jpg")
        #         cv2.imwrite(out_path, vis_img)

        else:
            empty_boxes = boxes_cxcywh[:0]
            if not requires_grad:
                empty_boxes = empty_boxes.detach()

            pseudo = {
                'image': images[b],
                'image_id': targets_meta[b]['image_id'],
                'orig_size': targets_meta[b]['orig_size'],
                'boxes': empty_boxes,
                'labels': logits_b.new_zeros((0,), dtype=torch.long).detach(),
                'scores': logits_b.new_zeros((0,)).detach(),
                'meta': meta,
            }
            pseudo_targets.append(pseudo)
    
    if as_batched_dict:
        batched = _transpose_pseudo_to_batched(pseudo_targets)
        if stack_tensors:
            batched = _pack_batched(batched)
        # 批处理完后再整体合入原始 outputs（不改变 outputs 本身），保持原始键名与批形状
        batched = _merge_batch_outputs(batched, outputs)
        return batched
    return pseudo_targets


def _transpose_pseudo_to_batched(pseudo_list: List[Dict]) -> Dict:
    """
    将 list[dict] 的伪标签列表转为按关键字聚合的批字典：
      { key: [val_img0, val_img1, ...] }，长度为 batch。
    注意：boxes/labels/scores 可能变长，保留为长度为B的列表（不做 padding）。
    """
    if len(pseudo_list) == 0:
        return {}
    keys = set()
    for d in pseudo_list:
        keys.update(d.keys())
    batched: Dict[str, List] = {}
    for k in keys:
        batched[k] = [d.get(k, None) for d in pseudo_list]
    return batched


def _pack_batched(batched: Dict[str, List]):
    """
    将按关键字聚合、长度为 B 的列表尝试堆叠为 [B, ...] 张量；
    - 若该键对应的 B 个元素均为 Tensor/Parameter 且形状/设备/dtype一致，则使用 torch.stack(dim=0)
    - 若均为 dict，则对其键做递归打包（仅对所有样本都存在的键）
    - 若均为 list/tuple 且长度一致，则逐位置递归打包
    - 其他情形保持为列表（变长对象、不一致类型等）
    """
    def same_shape_dtype_device(vals):
        import torch
        import torch.nn as nn
        if not all(isinstance(v, (torch.Tensor, nn.Parameter)) for v in vals):
            return False
        s0, d0, dev0 = vals[0].shape, vals[0].dtype, vals[0].device
        for v in vals[1:]:
            if v.shape != s0 or v.dtype != d0 or v.device != dev0:
                return False
        return True

    def can_pad_first_dim(vals):
        import torch
        import torch.nn as nn
        # 所有都是张量/参数，dtype/device一致，且除第一维外其余维度一致
        if not all(isinstance(v, (torch.Tensor, nn.Parameter)) for v in vals):
            return False
        d0, dev0 = vals[0].dtype, vals[0].device
        base_tail = vals[0].shape[1:]
        for v in vals[1:]:
            if v.dtype != d0 or v.device != dev0 or v.shape[1:] != base_tail:
                return False
        return True

    def pack(values):
        import torch
        import torch.nn as nn
        # 尝试张量打包
        if same_shape_dtype_device(values):
            return torch.stack(values, dim=0)
        # 尝试按第一维补齐后再打包
        if can_pad_first_dim(values):
            d0, dev0 = values[0].dtype, values[0].device
            tail = values[0].shape[1:]
            lens = [v.shape[0] for v in values]
            max_len = max(lens)
            if d0.is_floating_point:
                fill = 0.0
            else:
                # 整数类型（如 labels）用 -1 作为 padding
                fill = -1
            out = torch.full((len(values), max_len, *tail), fill_value=fill, dtype=d0, device=dev0)
            for i, v in enumerate(values):
                n = v.shape[0]
                if n > 0:
                    out[i, :n] = v
            return out
        # 全 dict：对共同键递归打包
        if all(isinstance(v, dict) for v in values):
            common_keys = set(values[0].keys())
            for v in values[1:]:
                common_keys &= set(v.keys())
            return {k: pack([v[k] for v in values]) for k in common_keys}
        # 全 list/tuple 且长度一致：逐位置打包
        if all(isinstance(v, (list, tuple)) for v in values):
            lens = [len(v) for v in values]
            if len(set(lens)) == 1:
                L = lens[0]
                return [pack([v[i] for v in values]) for i in range(L)]
        # 其他：保持列表
        return values

    packed = {}
    for k, vlist in batched.items():
        # 'meta' 保持原始每图列表形式，不参与打包/堆叠
        if k == 'meta':
            packed[k] = vlist
            continue
        packed[k] = pack(vlist)
    return packed

def _merge_batch_outputs(batched: Dict, outputs: Dict) -> Dict:
    """在批级别将原始 outputs 的键整体并入 batched，保持原始批形状；
    仅读取并 detach，不修改 outputs 本身。
    - Tensor/Parameter 且第一维为 B：直接 detach 后赋值
    - List/Tuple 且元素为 Tensor/Parameter 且第一维为 B：以同结构 list 存放，每个元素 detach
    - Dict：递归 detach 后放入
    - 包含全局键（'up','reg_scale','enc_meta'）也会并入
    若 batched 已有同名键（如由伪标签生成的 'boxes'），以 batched 为准，不覆盖
    """
    import torch
    import torch.nn as nn

    def detach_nested(x):
        if isinstance(x, (torch.Tensor, nn.Parameter)):
            return x.detach()
        if isinstance(x, list):
            return [detach_nested(e) for e in x]
        if isinstance(x, tuple):
            return tuple(detach_nested(e) for e in x)
        if isinstance(x, dict):
            return {k: detach_nested(v) for k, v in x.items()}
        return x

    B = None
    # 猜测批大小
    for v in outputs.values():
        if isinstance(v, (torch.Tensor, nn.Parameter)) and v.dim() >= 1:
            B = int(v.shape[0])
            break
        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (torch.Tensor, nn.Parameter)) and v[0].dim() >= 1:
            B = int(v[0].shape[0])
            break

    GLOBAL_INCLUDE_KEYS = { 'up', 'reg_scale', 'enc_meta', 'aux_outputs', 'enc_aux_outputs' }
    for k, v in outputs.items():
        # 全局键无条件并入（不覆盖已有键，但优先准备好待加入的值）
        if k in GLOBAL_INCLUDE_KEYS:
            val = detach_nested(v)
            if k not in batched:
                batched[k] = val
            continue
        if k in batched:
            continue
        to_add = None
        if isinstance(v, (torch.Tensor, nn.Parameter)):
            if B is not None and v.dim() >= 1 and v.shape[0] == B:
                to_add = v.detach()
        elif isinstance(v, (list, tuple)) and len(v) > 0:
            # 情形1：backbone 特征：list of Tensor，每个 [B,...]
            if all(isinstance(e, (torch.Tensor, nn.Parameter)) and e.dim() >= 1 and (B is None or e.shape[0] == B) for e in v):
                to_add = [e.detach() for e in v]
            # 情形2：list 长度等于 B，按样本分布的结构，直接整体并入
            elif B is not None and len(v) == B:
                to_add = detach_nested(v)
            # 情形3：list of dict（如 aux_outputs/enc_aux_outputs），递归整体并入
            elif any(isinstance(e, dict) for e in v):
                to_add = detach_nested(v)
        elif isinstance(v, dict):
            to_add = detach_nested(v)

        if to_add is not None:
            batched[k] = to_add
    return batched

