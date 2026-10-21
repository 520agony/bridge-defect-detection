r"""
_fusion_contrast.py — 纯图像估算 vs 点云融合 vs 真值 对照表
=============================================================
对 fusion_sim/sample02、sample03：
  YOLO(当前部署权重第一轮s)检出 -> 每检出同时算两条线：
  ① 纯图像线：crack长度=mask像素PCA极差×mpp；spall面积=mask像素数×mpp²
     （mpp=中位深度/fx；这是"不用点云时能做到的最好换算"）
  ② 融合线：points_in_mask->去噪->measure_length/area_3d（late_fusion）
  真值：truth.json
输出：TEMP/fusion_contrast.json + 终端对照表
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:\开始做项目\YOLOv26-Project")
SIM = Path(r"D:\开始做项目\fusion_sim")
OUT = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\fusion_contrast.json")
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO
from src.fusion.late_fusion import (load_camera, read_ply_ascii, points_in_mask,
                                    statistical_outlier_filter, measure_length_3d,
                                    measure_area_3d, median_depth, meters_per_pixel)


def mask_pixel_stats(mask_abs):
    """mask 绝对像素点列 -> (像素PCA极差, 像素点数)"""
    p = np.asarray(mask_abs, dtype=np.float64)
    if len(p) < 3:
        return 0.0, 0
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    proj = (p - c) @ vt[0]
    span = float(proj.max() - proj.min())
    # 像素面积：用多边形填充计数
    m = np.zeros((960, 1280), dtype=np.uint8)
    cv2.fillPoly(m, [p.astype(np.int32)], 1)
    return span, int(m.sum())


def main():
    model = YOLO(str(ROOT / "weights" / "bridge_defect_best.pt"))
    rows = []
    for sample in ("sample02", "sample03"):
        sdir = SIM / sample
        img = cv2.imdecode(np.fromfile(str(sdir / "rgb.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        K, R, t, _, _ = load_camera(str(sdir / "meta.json"))
        P, _ = read_ply_ascii(str(sdir / "cloud.ply"))
        truth = json.loads((sdir / "truth.json").read_text(encoding="utf-8"))
        r = model.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
        if r.masks is None:
            print(f"[{sample}] 无检出")
            continue
        for i, box in enumerate(r.boxes):
            cls = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            poly = (r.masks.xyn[i].reshape(-1, 2) * np.array([W, H])).tolist()
            idx = points_in_mask(P, K, R, t, poly, W, H)
            Pf = P[idx]
            keep = statistical_outlier_filter(Pf)
            Pf = Pf[keep]
            dep = median_depth(Pf, K, R, t) if len(Pf) else float(t[2])
            mpp = meters_per_pixel(dep, K)
            span_px, npix = mask_pixel_stats(poly)
            row = {"sample": sample, "class": cls, "conf": round(conf, 3),
                   "n_pts": len(Pf), "mpp": round(mpp, 6)}
            if cls == "crack":
                row["img_only_m"] = round(span_px * mpp, 4)
                row["fused_m"] = round(measure_length_3d(Pf), 4) if len(Pf) >= 2 else 0.0
                row["truth_m"] = truth.get("crack_length_m")
            else:
                row["img_only_sqm"] = round(npix * mpp * mpp, 6)
                row["fused_sqm"] = round(measure_area_3d(Pf), 6) if len(Pf) >= 3 else 0.0
                row["truth_sqm"] = truth.get("spall_area_sqm")
            rows.append(row)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {len(rows)} 条 -> {OUT}")
    for row in rows:
        if "truth_m" in row:
            ie = abs(row["img_only_m"] - row["truth_m"]) / row["truth_m"] * 100
            fe = abs(row["fused_m"] - row["truth_m"]) / row["truth_m"] * 100
            print(f"{row['sample']} crack@{row['conf']}: 纯图像 {row['img_only_m']}m( err {ie:.1f}% ) "
                  f"| 融合 {row['fused_m']}m( err {fe:.1f}% ) | 真值 {row['truth_m']}m")
        else:
            ie = abs(row["img_only_sqm"] - row["truth_sqm"]) / row["truth_sqm"] * 100
            fe = abs(row["fused_sqm"] - row["truth_sqm"]) / row["truth_sqm"] * 100
            print(f"{row['sample']} {row['class']}@{row['conf']}: 纯图像 {row['img_only_sqm']}㎡( err {ie:.1f}% ) "
                  f"| 融合 {row['fused_sqm']}㎡( err {fe:.1f}% ) | 真值 {row['truth_sqm']}㎡")


if __name__ == "__main__":
    main()
