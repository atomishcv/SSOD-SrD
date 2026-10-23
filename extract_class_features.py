import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

# 将工程根目录加入 sys.path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = FILE_DIR
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS
from engine.deim.box_ops import box_cxcywh_to_xyxy


def to_device_batch(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return [to_device_batch(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: to_device_batch(v, device) for k, v in obj.items()}
    return obj


@torch.no_grad()
def extract_encoder_features(model, images, targets, encoder_type='last'):
    """
    从 Encoder 提取每类特征。返回 {class_id: [C]} 字典。
    encoder_type: 'last' | 'multi'
    """
    net = model.module if hasattr(model, 'module') else model
    # 强制动态位置编码
    if hasattr(net, 'encoder') and hasattr(net.encoder, 'eval_spatial_size'):
        original_eval_size = net.encoder.eval_spatial_size
        net.encoder.eval_spatial_size = None
    else:
        original_eval_size = None

    try:
        feats = net.backbone(images)  # (c2,c3,c4)
        enc = net.encoder(feats) if hasattr(net, 'encoder') else feats
    finally:
        if original_eval_size is not None:
            net.encoder.eval_spatial_size = original_eval_size

    if isinstance(enc, (list, tuple)):
        if encoder_type == 'multi':
            target_size = enc[0].shape[-2:]
            up = []
            for f in enc:
                if f.shape[-2:] != target_size:
                    f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                up.append(f)
            enc_feat = torch.cat(up, dim=1)  # (B, C*, H, W)
        else:
            enc_feat = enc[-1]  # (B,C,H,W)
    else:
        enc_feat = enc

    B, C, H, W = enc_feat.shape
    out = defaultdict(list)
    for b in range(B):
        tgt = targets[b]
        if 'boxes' not in tgt or 'labels' not in tgt:
            continue
        boxes = tgt['boxes']
        labels = tgt['labels']
        if boxes.numel() == 0:
            continue
        f = enc_feat[b]  # (C,H,W)
        for box, lab in zip(boxes, labels):
            cid = int(lab.item())
            cx, cy, bw, bh = box.tolist()
            x1 = max(0, int((cx - bw / 2) * W))
            y1 = max(0, int((cy - bh / 2) * H))
            x2 = min(W, int((cx + bw / 2) * W))
            y2 = min(H, int((cy + bh / 2) * H))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = f[:, y1:y2, x1:x2]                   # (C,h,w)
            roi = F.adaptive_avg_pool2d(roi, (1, 1))   # (C,1,1)
            vec = roi.view(C)                          # (C,)
            vec = F.normalize(vec, p=2, dim=0)
            out[cid].append(vec.cpu())
    return out


def boxes_iou_xyxy(a, b):
    """
    a: (Na, 4) xyxy; b: (Nb, 4) xyxy
    return: (Na, Nb)
    """
    Na = a.shape[0]; Nb = b.shape[0]
    if Na == 0 or Nb == 0:
        return a.new_zeros((Na, Nb))
    lt = torch.max(a[:, None, :2], b[None, :, :2])   # (Na,Nb,2)
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])   # (Na,Nb,2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    union = area_a + area_b - inter + 1e-6
    return inter / union


@torch.no_grad()
def extract_decoder_features(model, images, targets, score_thresh=0.3, topk=1):
    """
    从 Decoder 提取每类特征。
      1) 前向: outputs = model(images, targets=None, is_teacher=True)
      2) 对每个 GT，用 IoU(+score) 匹配 query
      3) 取 dec_hs 最后一层对应 query 的隐向量
    返回: {class_id: [C]} 字典
    """
    net = model.module if hasattr(model, 'module') else model
    # 强制动态位置编码，避免尺寸不匹配
    original_eval_size = None
    if hasattr(net, 'encoder') and hasattr(net.encoder, 'eval_spatial_size'):
        original_eval_size = net.encoder.eval_spatial_size
        net.encoder.eval_spatial_size = None
    try:
        # 优先避免 DN：不带 targets
        outputs = None
        try:
            outputs = net(images, targets=None, is_teacher=True)
        except Exception:
            outputs = None
        # 若失败，再尝试带 targets 前向（部分实现需要 targets 才能返回 dec_hs）
        if outputs is None:
            try:
                outputs = net(images, targets=targets, is_teacher=True)
            except Exception:
                # 若仍失败，放弃本批次 decoder 特征，返回空
                return defaultdict(list)
    finally:
        if original_eval_size is not None:
            net.encoder.eval_spatial_size = original_eval_size
    if not isinstance(outputs, dict):
        return defaultdict(list)

    dec_hs = None
    if 'dec_hs' in outputs and isinstance(outputs['dec_hs'], dict):
        last_key = max(outputs['dec_hs'].keys())
        dec_hs = outputs['dec_hs'][last_key]  # (B,Q,C)
    elif hasattr(net, 'decoder') and hasattr(net.decoder, 'last_hidden_state'):
        dec_hs = net.decoder.last_hidden_state
    if dec_hs is None:
        return defaultdict(list)

    pred_boxes = outputs.get('pred_boxes', None)    # (B,Q,4) cxcywh
    pred_logits = outputs.get('pred_logits', None)  # (B,Q,num_classes)
    if pred_boxes is None or pred_logits is None:
        return defaultdict(list)

    B, Q, C = dec_hs.shape
    out = defaultdict(list)
    for b in range(B):
        tgt = targets[b]
        if 'boxes' not in tgt or 'labels' not in tgt:
            continue
        gt_boxes = tgt['boxes']   # (Ng,4) cxcywh
        gt_labels = tgt['labels']
        if gt_boxes.numel() == 0:
            continue

        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)   # (Ng,4)
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])  # (Q,4)
        probs = pred_logits[b].softmax(-1)       # (Q, num_classes)
        ious = boxes_iou_xyxy(gt_xyxy, pred_xyxy)      # (Ng, Q)

        for gi in range(gt_xyxy.shape[0]):
            cid = int(gt_labels[gi].item())
            # 分数掩码
            score_mask = probs[:, cid] >= score_thresh if cid < probs.shape[1] else torch.zeros(Q, dtype=torch.bool, device=probs.device)
            iou_row = ious[gi]
            iou_row = iou_row.masked_fill(~score_mask, -1.0) if score_thresh > 0 else iou_row
            k = min(topk, Q)
            topk_iou, topk_idx = torch.topk(iou_row, k=k)
            valid = topk_iou > 0
            if not valid.any():
                # 退回 IoU Top-1
                topk_idx = torch.argmax(ious[gi]).view(1)
                valid = torch.tensor([True], device=topk_idx.device)
            for q_idx, ok in zip(topk_idx.tolist(), valid.tolist()):
                if not ok:
                    continue
                vec = dec_hs[b, q_idx]  # (C,)
                vec = F.normalize(vec, p=2, dim=0)
                out[cid].append(vec.cpu())
    return out


def reduce_and_stats(feat_dict):
    """
    将 {cid: [vec...]} 计算为原型与统计
    返回: prototypes, stats，其中 prototypes[cid]=Tensor(C,)
    """
    prototypes = {}
    stats = {}
    for cid, lst in feat_dict.items():
        if len(lst) == 0:
            continue
        mat = torch.stack(lst)  # (N,C)
        proto = F.normalize(mat.mean(dim=0), p=2, dim=0)
        prototypes[cid] = proto
        stats[cid] = {
            'count': len(lst),
            'std': mat.std(dim=0).mean().item(),
            'dim': mat.shape[1],
        }
    return prototypes, stats


def main():
    parser = argparse.ArgumentParser(description='提取每个类别的 Encoder 与 Decoder 特征（有监督数据）')
    parser.add_argument('--config', type=str, default='configs/deimv2_ssod/deimv2_dinov3_x_coco.yml', help='配置文件路径，如 configs/deimv2_ssod/deimv2_dinov3_x_coco.yml')
    parser.add_argument('--checkpoint', type=str, default='/data/DEIMv2/outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth', help='加载已训练权重（可选）')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--encoder-type', type=str, default='multi', choices=['last', 'multi'],
                        help='Encoder 特征类型：last=最后一层；multi=多层上采样拼接')
    parser.add_argument('--score-thresh', type=float, default=0.2, help='Decoder 匹配的类别分数阈值')
    parser.add_argument('--topk', type=int, default=1, help='每个 GT 选择的 query 数（Decoder）')
    parser.add_argument('--max-images', type=int, default=200, help='最多处理多少张训练图（None=全量）')
    parser.add_argument('--max-samples-per-class', type=int, default=200,
                        help='每类最多保留多少个样本向量（避免过大内存）')
    parser.add_argument('--output', type=str, default='outputs/class_features/coco_enc_dec.pth', help='保存路径')
    parser.add_argument('--aggregate-mode', type=str, default='mean', choices=['mean', 'samples'],
                        help="聚合方式：'mean' 在线更新每类均值（重复时做平均）；'samples' 保存样本列表，末尾再求均值")
    args = parser.parse_args()

    # 准备配置与 solver
    update_dict = yaml_utils.parse_cli(None)
    cfg = YAMLConfig(args.config, **(update_dict or {}))

    task_name = cfg.yaml_cfg.get('task', 'detection')
    if task_name not in TASKS:
        task_name = 'detection'
    solver = TASKS[task_name](cfg)
    solver.train()  # 初始化 dataloader、optimizer、scaler 等

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    solver.model.to(device)
    solver.model.eval()

    # 加载检查点
    if args.checkpoint:
        print(f'加载检查点: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state = ckpt.get('model', ckpt)
        model_keys = set(solver.model.state_dict().keys())
        ckpt_keys = set(state.keys())
        model_has_module = any(k.startswith('module.') for k in model_keys)
        ckpt_has_module = any(k.startswith('module.') for k in ckpt_keys)
        new_state = {}
        if ckpt_has_module and not model_has_module:
            for k, v in state.items():
                new_state[k[7:] if k.startswith('module.') else k] = v
        elif not ckpt_has_module and model_has_module:
            for k, v in state.items():
                new_state[f'module.{k}'] = v
        else:
            new_state = state
        missing, unexpected = solver.model.load_state_dict(new_state, strict=False)
        if missing:
            print(f'⚠️ Missing keys: {missing[:10]}{"..." if len(missing)>10 else ""}')
        if unexpected:
            print(f'⚠️ Unexpected keys: {unexpected[:10]}{"..." if len(unexpected)>10 else ""}')

    # 选择聚合策略
    aggregate_mean = (args.aggregate_mode == 'mean')
    encoder_bank = defaultdict(list) if not aggregate_mean else None
    decoder_bank = defaultdict(list) if not aggregate_mean else None
    encoder_mean = {} if aggregate_mean else None        # {cid: tensor(C,)}
    decoder_mean = {} if aggregate_mean else None
    encoder_count = defaultdict(int) if aggregate_mean else None
    decoder_count = defaultdict(int) if aggregate_mean else None

    def update_running_mean(mean_dict, count_dict, cid, vec):
        if cid not in mean_dict:
            mean_dict[cid] = vec.clone().cpu()
            count_dict[cid] = 1
        else:
            n = count_dict[cid]
            new_mean = (mean_dict[cid] * n + vec.cpu()) / (n + 1)
            mean_dict[cid] = F.normalize(new_mean, p=2, dim=0)
            count_dict[cid] = n + 1

    total_imgs = 0
    for step, (images, targets) in enumerate(solver.train_dataloader):
        if args.max_images is not None and total_imgs >= args.max_images:
            break
        total_imgs += images.shape[0]

        images = images.to(device)
        targets = to_device_batch(targets, device)

        with torch.no_grad():
            # Encoder
            enc_dict = extract_encoder_features(
                solver.model, images, targets, encoder_type=args.encoder_type
            )
            for cid, lst in enc_dict.items():
                if len(lst) == 0:
                    continue
                if aggregate_mean:
                    for v in lst:
                        update_running_mean(encoder_mean, encoder_count, cid, v)
                else:
                    remain = args.max_samples_per_class - len(encoder_bank[cid])
                    if remain > 0:
                        encoder_bank[cid].extend(lst[:remain])

            # Decoder
            dec_dict = extract_decoder_features(
                solver.model, images, targets, score_thresh=args.score_thresh, topk=args.topk
            )
            for cid, lst in dec_dict.items():
                if len(lst) == 0:
                    continue
                if aggregate_mean:
                    for v in lst:
                        update_running_mean(decoder_mean, decoder_count, cid, v)
                else:
                    remain = args.max_samples_per_class - len(decoder_bank[cid])
                    if remain > 0:
                        decoder_bank[cid].extend(lst[:remain])

        if step % 20 == 0:
            print(f'Processed steps: {step}, total images: {total_imgs}')

    # 计算原型与统计
    def reduce_and_stats_local(bank):
        protos = {}
        stats = {}
        for cid, lst in bank.items():
            if len(lst) == 0:
                continue
            mat = torch.stack(lst)
            proto = F.normalize(mat.mean(dim=0), p=2, dim=0)
            protos[cid] = proto
            stats[cid] = {'count': len(lst), 'std': mat.std(dim=0).mean().item(), 'dim': mat.shape[1]}
        return protos, stats

    if aggregate_mean:
        enc_protos = encoder_mean
        dec_protos = decoder_mean
        enc_stats = {cid: {'count': encoder_count[cid], 'std': 0.0, 'dim': enc_protos[cid].numel()}
                     for cid in enc_protos.keys()}
        dec_stats = {cid: {'count': decoder_count[cid], 'std': 0.0, 'dim': dec_protos[cid].numel()}
                     for cid in dec_protos.keys()}
    else:
        enc_protos, enc_stats = reduce_and_stats_local(encoder_bank)
        dec_protos, dec_stats = reduce_and_stats_local(decoder_bank)

    if aggregate_mean:
        all_cids = sorted(set(list(enc_protos.keys()) + list(dec_protos.keys())))
    else:
        all_cids = sorted(set(list(encoder_bank.keys()) + list(decoder_bank.keys())))
    merged_stats = {}
    for cid in all_cids:
        merged_stats[cid] = {
            'encoder_count': (encoder_count[cid] if aggregate_mean else len(encoder_bank.get(cid, []))),
            'decoder_count': (decoder_count[cid] if aggregate_mean else len(decoder_bank.get(cid, []))),
            'encoder_dim': enc_protos[cid].numel() if cid in enc_protos else 0,
            'decoder_dim': dec_protos[cid].numel() if cid in dec_protos else 0,
            'encoder_std': enc_stats.get(cid, {}).get('std', 0.0),
            'decoder_std': dec_stats.get(cid, {}).get('std', 0.0),
        }

    # 保存
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if aggregate_mean:
        save_obj = {
            'encoder_prototypes': enc_protos,
            'decoder_prototypes': dec_protos,
            'stats': merged_stats,
            'meta': {
                'config': args.config,
                'checkpoint': args.checkpoint,
                'encoder_type': args.encoder_type,
                'score_thresh': args.score_thresh,
                'topk': args.topk,
                'max_images': args.max_images,
                'max_samples_per_class': args.max_samples_per_class,
                'aggregate_mode': args.aggregate_mode,
            }
        }
    else:
        save_obj = {
            'encoder_samples': {cid: (torch.stack(v) if len(v) > 0 else torch.empty(0))
                                for cid, v in encoder_bank.items()},
            'decoder_samples': {cid: (torch.stack(v) if len(v) > 0 else torch.empty(0))
                                for cid, v in decoder_bank.items()},
            'encoder_prototypes': enc_protos,
            'decoder_prototypes': dec_protos,
            'stats': merged_stats,
            'meta': {
                'config': args.config,
                'checkpoint': args.checkpoint,
                'encoder_type': args.encoder_type,
                'score_thresh': args.score_thresh,
                'topk': args.topk,
                'max_images': args.max_images,
                'max_samples_per_class': args.max_samples_per_class,
                'aggregate_mode': args.aggregate_mode,
            }
        }
    torch.save(save_obj, out_path)
    print(f'✅ 已保存类别特征到: {out_path}')
    print(f'  - 类别数 (encoder): {len(enc_protos)}')
    print(f'  - 类别数 (decoder): {len(dec_protos)}')


if __name__ == '__main__':
    # 单卡运行，用于离线特征提取
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    main()


