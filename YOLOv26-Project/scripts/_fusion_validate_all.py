r"""
_fusion_validate_all.py — B2 全场景验证（双权重 × 三病害）
============================================================
权重：distill（第三轮，仿真工作基准）+ s_new（第一轮，当前部署）
场景：sample01..sample10（有 truth.json 的）
每场景每权重：YOLO检出 -> 融合测量 -> 对真值误差；同时记检出率
病害映射：crack->长度；spalling/hole->面积（hole 额外记深度可检性）
输出：TEMP/fusion_all.json + 终端汇总表
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:\开始做项目\YOLOv26-Project")
SIM = Path(r"D:\开始做项目\fusion_sim")
OUT = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\fusion_all.json")
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO
from src.fusion.late_fusion import (load_camera, read_ply_ascii, points_in_mask,
                                    statistical_outlier_filter, measure_length_3d,
                                    measure_area_3d, median_depth, meters_per_pixel)

WEIGHTS = {
    "distill": r"D:\开始做项目\第三轮\best.pt",
    "s_new": str(ROOT / "weights" / "bridge_defect_best.pt"),
}
TRUTH_KEY = {"crack": "crack_length_m", "spalling": "spall_area_sqm", "hole": "hole_area_sqm"}


def main():
    models = {k: YOLO(v) for k, v in WEIGHTS.items()}
    samples = sorted([d for d in SIM.iterdir()
                      if d.is_dir() and (d / "truth.json").exists()], key=lambda p: p.name)
    recs = []
    for sdir in samples:
        img = cv2.imdecode(np.fromfile(str(sdir / "rgb.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        K, R, t, _, _ = load_camera(str(sdir / "meta.json"))
        P, _ = read_ply_ascii(str(sdir / "cloud.ply"))
        truth = json.loads((sdir / "truth.json").read_text(encoding="utf-8"))
        for wtag, model in models.items():
            r = model.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
            dets = []
            if r.masks is not None:
                for i, box in enumerate(r.boxes):
                    dets.append((r.names[int(box.cls[0])], float(box.conf[0]),
                                 (r.masks.xyn[i].reshape(-1, 2) * np.array([W, H])).tolist()))
            for cls, conf, poly in dets:
                if cls not in TRUTH_KEY or TRUTH_KEY[cls] not in truth:
                    continue
                idx = points_in_mask(P, K, R, t, poly, W, H)
                Pf = P[idx]
                Pf = Pf[statistical_outlier_filter(Pf)]
                dep = median_depth(Pf, K, R, t) if len(Pf) else float(t[2])
                mpp = meters_per_pixel(dep, K)
                tv = truth[TRUTH_KEY[cls]]
                if cls == "crack":
                    fv = round(measure_length_3d(Pf), 4) if len(Pf) >= 2 else 0.0
                    p = np.asarray(poly)
                    c = p.mean(axis=0)
                    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
                    proj = (p - c) @ vt[0]
                    iv = round(float(proj.max() - proj.min()) * mpp, 4)
                else:
                    fv = round(measure_area_3d(Pf), 6) if len(Pf) >= 3 else 0.0
                    m = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(m, [np.asarray(poly).astype(np.int32)], 1)
                    iv = round(int(m.sum()) * mpp * mpp, 6)
                recs.append({"sample": sdir.name, "weight": wtag, "class": cls,
                             "conf": round(conf, 3), "n_pts": len(Pf),
                             "img_only": iv, "fused": fv, "truth": tv,
                             "err_img": round(abs(iv - tv) / tv * 100, 1),
                             "err_fused": round(abs(fv - tv) / tv * 100, 1)})
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    # 汇总
    print(f"[OK] {len(recs)} 条 -> {OUT}")
    print(f"{'sample':<10}{'w':<8}{'class':<10}{'conf':<7}{'img_err%':<9}{'fused_err%':<11}truth")
    for r in recs:
        print(f"{r['sample']:<10}{r['weight']:<8}{r['class']:<10}{r['conf']:<7}"
              f"{r['err_img']:<9}{r['err_fused']:<11}{r['truth']}")
    # 按权重×病害聚合
    from collections import defaultdict
    agg = defaultdict(list)
    for r in recs:
        agg[(r["weight"], r["class"])].append(r["err_fused"])
    print("\n== 融合误差均值（权重×病害） ==")
    for k in sorted(agg):
        v = agg[k]
        print(f"  {k[0]:<8}{k[1]:<10}n={len(v)} mean={sum(v)/len(v):.1f}% max={max(v):.1f}%")


if __name__ == "__main__":
    main()
