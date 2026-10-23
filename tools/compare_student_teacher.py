#!/usr/bin/env python3
"""
比较同一张图片在学生模型与教师（EMA）模型上的预测差异。

使用方式：
    python tools/compare_student_teacher.py \
        --config configs/deimv2_ssod/deimv2_dinov3_x_coco.yml \
        --checkpoint outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth \
        --image path/to/image.jpg \
        --output-dir visualizations/student_teacher_compare
"""

import argparse
import os
import sys
from pathlib import Path
import copy

import cv2
import torch
import torchvision.transforms as T

# 将项目根目录加入 PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.core import YAMLConfig
from engine.optim import ModelEMA
from engine.solver.pseudo_label_generator import tensor_to_bgr, draw_predictions


def preprocess_image(image_path: Path, input_size: int = 640):
    """与 test_single_image 中保持一致的预处理：缩放 + padding + 归一化。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    orig_h, orig_w = image.shape[:2]
    scale = min(input_size / orig_h, input_size / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = torch.zeros((input_size, input_size, 3), dtype=torch.uint8).numpy()
    pad_top = (input_size - new_h) // 2
    pad_left = (input_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    # OpenCV -> PIL 风格转换
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil = T.ToPILImage()(rgb)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(pil).unsqueeze(0)  # [1,3,H,W]

    return tensor, image, scale, (pad_left, pad_top)


def _parse_device_ids(device_ids_str: str):
    ids = []
    for part in device_ids_str.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            raise ValueError(f"非法 GPU ID: {part}")
    if not ids:
        raise ValueError("device-ids 至少需要包含一个有效的 GPU 编号")
    return ids


def load_student_and_teacher(cfg: YAMLConfig,
                             checkpoint_path: Path,
                             device: torch.device,
                             device_ids,
                             multi_gpu: bool):

    def _align_prefix(state_dict, target_has_module_prefix: bool):
        """对齐 DDP 的 'module.' 前缀"""
        if not state_dict:
            return state_dict
        sd_has_prefix = next(iter(state_dict)).startswith("module.")
        if sd_has_prefix != target_has_module_prefix:
            state_dict = {
                (k[7:] if sd_has_prefix else f"module.{k}"): v
                for k, v in state_dict.items()
            }
        return state_dict

    # 读取 ckpt
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in ckpt:
        raise RuntimeError(f"checkpoint 中不含 'model' 权重：{checkpoint_path}")
    state_dict_model = ckpt["model"]

    student_base = copy.deepcopy(cfg.model).to(device)
    if multi_gpu:
        student_model = torch.nn.DataParallel(student_base, device_ids=device_ids,
                                              output_device=device.index)
    else:
        student_model = student_base
    sd_student = _align_prefix(state_dict_model, next(iter(student_model.state_dict())).startswith("module."))
    ret_s = student_model.load_state_dict(sd_student, strict=False)
    print(f"[student<-model] missing={len(ret_s.missing_keys)} unexpected={len(ret_s.unexpected_keys)}")
    student_model.eval()

    # ===== 教师模型（同样用 model 权重）=====
    teacher_base = copy.deepcopy(cfg.model).to(device)
    if multi_gpu:
        teacher_model = torch.nn.DataParallel(teacher_base, device_ids=device_ids,
                                              output_device=device.index)
    else:
        teacher_model = teacher_base
    sd_teacher = _align_prefix(state_dict_model, next(iter(teacher_model.state_dict())).startswith("module."))
    ret_t = teacher_model.load_state_dict(sd_teacher, strict=False)
    print(f"[teacher<-model] missing={len(ret_t.missing_keys)} unexpected={len(ret_t.unexpected_keys)}")
    teacher_model.eval()

    # 日志里带上指标，方便确认加载的是同一份 ckpt
    print(f"ckpt metrics -> ap: {ckpt.get('ap', None)}, best_ap: {ckpt.get('best_ap', None)}")

    return student_model, teacher_model


@torch.no_grad()
def run_inference(model, image_tensor, is_teacher: bool):
    """执行一次前向传播，返回预测 logits/boxes。"""
    outputs = model(image_tensor, is_teacher=is_teacher)
    return outputs["pred_logits"], outputs["pred_boxes"]


def extract_predictions(pred_logits: torch.Tensor,
                        pred_boxes: torch.Tensor,
                        conf_thresh: float = 0.3):
    """将模型输出转换为可视化的 boxes/labels/scores。"""
    probs = pred_logits.sigmoid()
    scores, labels = probs.max(dim=-1)
    keep = scores >= conf_thresh
    boxes = pred_boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    return {
        "boxes": boxes.cpu(),
        "labels": labels.cpu(),
        "scores": scores.cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="比较学生模型与 EMA 教师模型的预测差异")
    parser.add_argument("--config", type=str, default="configs/deimv2_ssod/deimv2_dinov3_x_coco.yml", help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="outputs/deimv2_dinov3_x_coco/checkpoint_best_ap0.4903_epoch051.pth", help="checkpoint 路径")
    parser.add_argument("--image", type=str, default="visualizations/ssod_debug/epoch_000/unsup_teacher_aug/batch0000_weak_0.jpg", help="测试图片路径")
    parser.add_argument("--output-dir", type=str, default="visualizations/student_teacher_compare",
                        help="输出可视化目录")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-ids", type=str, default="1,2",
                        help="逗号分隔的 GPU ID 列表，例如 '0,1'")
    parser.add_argument("--conf-thresh", type=float, default=0.3, help="可视化阈值")
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()

    device = torch.device(args.device)

    device_ids = _parse_device_ids(args.device_ids)
    if torch.cuda.is_available():
        primary_gpu = device_ids[0]
        torch.cuda.set_device(primary_gpu)
        device = torch.device(f"cuda:{primary_gpu}")
    else:
        device = torch.device("cpu")

    # 预处理图片（先在 CPU 上完成，便于决定是否使用多卡）
    image_tensor, orig_image, scale, pad_info = preprocess_image(Path(args.image), args.input_size)
    batch_size = image_tensor.size(0)
    use_data_parallel = (
        torch.cuda.is_available()
        and len(device_ids) > 1
        and batch_size >= len(device_ids)
    )
    if torch.cuda.is_available() and len(device_ids) > 1 and batch_size < len(device_ids):
        print(f"⚠️  batch_size={batch_size} 小于 GPU 数 {len(device_ids)}，自动降级为单卡推理。")

    cfg = YAMLConfig(args.config)
    student_model, teacher_model = load_student_and_teacher(
        cfg, Path(args.checkpoint), device, device_ids, use_data_parallel
    )

    # 准备输入图片
    image_tensor = image_tensor.to(device)

    # 推理：学生（强增强视角）
    student_logits, student_boxes = run_inference(student_model, image_tensor, is_teacher=False)
    student_pred = extract_predictions(student_logits[0], student_boxes[0], args.conf_thresh)

    # 推理：教师（弱增强视角）
    teacher_logits, teacher_boxes = run_inference(teacher_model, image_tensor, is_teacher=True)
    teacher_pred = extract_predictions(teacher_logits[0], teacher_boxes[0], args.conf_thresh)

    # 输出目录
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存对比结果
    student_vis = draw_predictions(orig_image, student_pred, scale, pad_info,
                                   conf_thresh=args.conf_thresh, input_size=args.input_size, i=0)
    teacher_vis = draw_predictions(orig_image, teacher_pred, scale, pad_info,
                                   conf_thresh=args.conf_thresh, input_size=args.input_size, i=1)

    cv2.imwrite(str(out_dir / "student_result.jpg"), student_vis)
    cv2.imwrite(str(out_dir / "teacher_result.jpg"), teacher_vis)

    print("\n=== 预测统计 ===")
    print(f"学生模型：{len(student_pred['boxes'])} 个框")
    print(f"教师模型：{len(teacher_pred['boxes'])} 个框")
    print(f"结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()

