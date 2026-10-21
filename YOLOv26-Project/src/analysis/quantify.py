"""
病害量化分析 CLI
================
检测 → 测量 → 等级评估 → 报告（JSON + Markdown）。

用法:
    ./.venv/Scripts/python.exe -m src.analysis.quantify \
        --source <图片或目录> --weights experiments/yolov26_baseline_v2/weights/best.pt \
        [--px-per-mm 10.0] [--conf 0.25] [--out reports/] [--limit N]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.analysis import CLASS_ZH
from src.analysis.grade import grade_defect
from src.analysis.measure import measure_defect
from src.analysis.report import build_report, render_markdown


def list_images(source):
    src = Path(source)
    if src.is_dir():
        return [p for p in sorted(src.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    return [src]


def process_image(model, img_path, conf, px_per_mm):
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    H, W = img.shape[:2]
    results = model.predict(img, conf=conf, imgsz=640, verbose=False)
    r = results[0]

    defects = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id]
        bbox = [x1, y1, x2, y2]
        meas = measure_defect(cls_name, img, bbox)
        grade = grade_defect(cls_name, meas, px_per_mm=px_per_mm)
        defects.append({
            "class": cls_name,
            "class_id": cls_id,
            "zh": CLASS_ZH.get(cls_name, cls_name),
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 1) for v in bbox],
            "measurements": meas,
            "grade": grade,
        })

    defects.sort(key=lambda d: d["grade"]["grade"], reverse=True)
    return build_report(img_path, (W, H), defects, px_per_mm=px_per_mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--px-per-mm", type=float, default=None, help="像素/毫米比例，提供则输出正式毫米级等级")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    imgs = list_images(args.source)
    if args.limit:
        imgs = imgs[: args.limit]
    if not imgs:
        raise SystemExit("未找到图片")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in imgs:
        try:
            report = process_image(model, img_path, args.conf, args.px_per_mm)
            stem = img_path.stem
            md_path = out_dir / f"{stem}_report.md"
            json_path = out_dir / f"{stem}_report.json"
            md_path.write_text(render_markdown(report), encoding="utf-8")
            json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] {img_path.name}: {report['total_defects']} 处病害, 最高 {report['max_grade']} 级 -> {md_path.name}")
        except Exception as e:
            print(f"[ERR] {img_path.name}: {e}")

    print(f"\n完成，报告保存在: {out_dir}")


if __name__ == "__main__":
    main()
