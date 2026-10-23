"""
将 mengu 目录下 LabelMe 格式标注 (per-image JSON) 转换为 COCO 格式标注 (单个 JSON)
"""

import json
import glob
import os

MENGU_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(MENGU_DIR, "annotations_coco.json")

CATEGORY_MAP = {"target": 1}

coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "target", "supercategory": "object"}],
}

ann_id = 1
json_files = sorted(glob.glob(os.path.join(MENGU_DIR, "*.json")))

for img_id, jf in enumerate(json_files, start=1):
    if os.path.basename(jf) == "annotations_coco.json":
        continue

    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth", 1024)
    img_h = data.get("imageHeight", 1024)

    jpg_base = os.path.splitext(os.path.basename(jf))[0] + ".jpg"
    file_name = jpg_base

    coco["images"].append({
        "id": img_id,
        "file_name": file_name,
        "width": img_w,
        "height": img_h,
    })

    for shape in data.get("shapes", []):
        label = shape["label"]
        cat_id = CATEGORY_MAP.get(label)
        if cat_id is None:
            print(f"  跳过未知类别: {label} (in {jf})")
            continue

        pts = shape["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0:
            continue

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0,
        })
        ann_id += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False, indent=2)

print(f"转换完成: {len(coco['images'])} 张图片, {len(coco['annotations'])} 个标注")
print(f"输出文件: {OUTPUT_JSON}")
