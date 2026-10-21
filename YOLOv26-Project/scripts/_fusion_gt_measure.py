r"""
_fusion_gt_measure.py — 真值几何直喂融合模块（测"量得准不准"，与检出解耦）
==========================================================================
每场景已知世界几何 -> 投影成像素mask -> points_in_mask -> 去噪 -> 量算 -> 对真值
crack: 矩形(x0..x1, ±20mm)四角投影；spall/hole: 圆周16点投影
输出：TEMP/fusion_gt.json + 终端表
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

SIM = Path(r"D:\开始做项目\fusion_sim")
OUT = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\fusion_gt.json")
sys.path.insert(0, r"D:\开始做项目\YOLOv26-Project")
from src.fusion.late_fusion import (load_camera, read_ply_ascii, points_in_mask,
                                    statistical_outlier_filter, measure_length_3d,
                                    measure_area_3d)

# 世界几何（与仿真脚本参数一致；01-03 为 v3 写死值）
GEO = {
    "sample01": {"crack": (-0.5, 0.5), "spall": None, "hole": None},  # 旧版场景，剥落几何未知，spall不统计
    "sample02": {"crack": (-0.5, 0.5), "spall": ((0.35, -0.25), 0.12), "hole": None},
    "sample03": {"crack": (-0.5, 0.5), "spall": ((0.35, -0.25), 0.12), "hole": None},
    "sample04": {"crack": (-0.3, 0.5), "spall": ((0.30, -0.20), 0.10), "hole": ((-0.25, 0.18), 0.09)},
    "sample05": {"crack": (-0.45, 0.55), "spall": ((-0.30, 0.22), 0.13), "hole": ((0.35, -0.15), 0.11)},
    "sample06": {"crack": (-0.1, 0.5), "spall": ((0.15, 0.25), 0.09), "hole": ((-0.35, -0.22), 0.07)},
    "sample07": {"crack": (-0.55, 0.55), "spall": ((0.40, 0.10), 0.11), "hole": ((0.05, -0.28), 0.10)},
    "sample08": {"crack": (0.0, 0.7), "spall": ((-0.25, -0.10), 0.14), "hole": ((0.30, 0.22), 0.12)},
    "sample09": {"crack": (-0.4, 0.5), "spall": ((0.20, -0.25), 0.08), "hole": ((-0.15, 0.20), 0.08)},
    "sample10": {"crack": (-0.5, 0.5), "spall": ((0.05, 0.28), 0.12), "hole": ((-0.30, 0.05), 0.11)},
}
TKEY = {"crack": "crack_length_m", "spall": "spall_area_sqm", "hole": "hole_area_sqm"}


def project(pts3, K, R, t):
    P = np.asarray(pts3, dtype=np.float64)
    Xc = (np.asarray(R) @ P.T).T + np.asarray(t)
    uv = (np.asarray(K) @ Xc.T).T[:, :2] / Xc[:, 2:3]
    return uv


def main():
    recs = []
    for sample, g in GEO.items():
        sdir = SIM / sample
        K, R, t, W, H = load_camera(str(sdir / "meta.json"))
        P, _ = read_ply_ascii(str(sdir / "cloud.ply"))
        truth = json.loads((sdir / "truth.json").read_text(encoding="utf-8"))
        masks = {}
        x0, x1 = g["crack"]
        masks["crack"] = [(x0, -0.02, 0), (x1, -0.02, 0), (x1, 0.02, 0), (x0, 0.02, 0)]
        for k in ("spall", "hole"):
            if g[k] is None:
                continue
            (cx, cy), r = g[k]
            masks[k] = [(cx + r * math.cos(a), cy + r * math.sin(a), 0)
                        for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
        for cls, pts3 in masks.items():
            if TKEY[cls] not in truth:
                continue
            poly = project(pts3, K, R, t).tolist()
            idx = points_in_mask(P, K, R, t, poly, W, H)
            Pf = P[idx]
            Pf = Pf[statistical_outlier_filter(Pf)]
            tv = truth[TKEY[cls]]
            fv = (measure_length_3d(Pf) if cls == "crack" else measure_area_3d(Pf)) if len(Pf) >= 3 else 0.0
            recs.append({"sample": sample, "class": cls, "n_pts": len(Pf),
                         "fused": round(float(fv), 6), "truth": tv,
                         "err": round(abs(float(fv) - tv) / tv * 100, 1)})
    OUT.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {len(recs)} 条 -> {OUT}")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in recs:
        agg[r["class"]].append(r["err"])
        print(f"  {r['sample']:<10}{r['class']:<9}n_pts={r['n_pts']:<6}fused={r['fused']:<10}truth={r['truth']:<10}err={r['err']}%")
    print("== 真值mask测量误差均值 ==")
    for k in sorted(agg):
        v = agg[k]
        print(f"  {k:<9}n={len(v)} mean={sum(v)/len(v):.1f}% max={max(v):.1f}%")


if __name__ == "__main__":
    main()
