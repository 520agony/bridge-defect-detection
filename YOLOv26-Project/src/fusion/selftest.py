"""
后融合自检：用已知真值的仿真场景验证整条链路
==============================================
场景：相机在原点看 +Z，5m 处有一面墙；
墙上有裂缝（真长 ~0.5025m）+ 剥落方块（真面积 0.06 m²）。
跑法：.venv/Scripts/python.exe src/fusion/selftest.py
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fusion.late_fusion import (  # noqa: E402
    fit_plane,
    measure_area_3d,
    measure_crack_fused_depth,
    median_depth,
    meters_per_pixel,
    points_in_mask,
    project_points,
    read_ply_ascii,
    statistical_outlier_filter,
)

rng = np.random.default_rng(42)
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'OK' if cond else 'FAIL'}] {name} {detail}")


# ---------- 仿真场景 ----------
W, H = 640, 480
K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], float)
R = np.eye(3)
t = np.zeros(3)

# 墙：z=5，3.6m × 2.6m，1cm 网格 + 2mm 噪声
gx, gy = np.meshgrid(np.arange(-1.8, 1.801, 0.01), np.arange(-1.3, 1.301, 0.01))
wall = np.stack([gx.ravel(), gy.ravel(),
                 5 + rng.normal(0, 0.002, gx.size)], axis=1)

# 裂缝：A(-0.25,0,5) -> B(0.25,0.05,5)，下沉 8mm 沟槽
A = np.array([-0.25, 0.0, 5 - 0.008])
B = np.array([0.25, 0.05, 5 - 0.008])
TRUE_LEN = float(np.linalg.norm(B - A))
s = np.linspace(0, 1, 400)[:, None]
crack = A[None, :] + s * (B - A)[None, :] + rng.normal(0, 0.001, (400, 3))

# 剥落：0.3 × 0.2m 方块，中心 (0.5,-0.3,5)
px, py = np.meshgrid(np.arange(0.35, 0.651, 0.01), np.arange(-0.4, -0.199, 0.01))
spall = np.stack([px.ravel(), py.ravel(),
                  5 + rng.normal(0, 0.0015, px.size)], axis=1)
TRUE_AREA = 0.3 * 0.2

P = np.vstack([wall, crack, spall])
print(f"场景点数: wall={len(wall)} crack={len(crack)} spall={len(spall)} 真长={TRUE_LEN:.4f}m 真面积={TRUE_AREA:.4f}sqm")

# ---------- 1. 投影 ----------
uv, z, valid = project_points(P, K, R, t)
check("投影有效性", valid.all(), f"valid={valid.sum()}/{len(P)}")
check("墙中心投影", True, f"uv[0]={uv[0].round(1)} (期望约[320,240]附近)")
cx = uv[len(wall) // 2]
check("中心像素≈(320,240)", abs(cx[0] - 320) < 8 and abs(cx[1] - 240) < 8, f"got {cx.round(1)}")

behind = np.array([[0, 0, -1.0]])
_, _, vb = project_points(behind, K, R, t)
check("相机后方点被剔除", not vb[0])

# ---------- 2. 模拟 YOLO mask（裂缝投影多边形）→ 筛点 ----------
uv_crack, _, _ = project_points(crack, K, R, t)
x1, y1 = uv_crack.min(axis=0) - 3
x2, y2 = uv_crack.max(axis=0) + 3
mask_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
idx = points_in_mask(P, K, R, t, mask_poly, W, H)
check("mask筛点命中裂缝", len(idx) > 300, f"选中{len(idx)}点")

# 沟槽提取：拟合平面，保留下沉超 3mm 的点（裂缝真实做法）
sel = P[idx]
n, c, _, _ = fit_plane(sel)
signed = (sel - c) @ n
if n[2] > 0:
    signed = -signed  # 法向朝相机为正凹陷为负，统一 recessed<0
groove = sel[signed < -0.003]
check("沟槽提取", len(groove) > 200, f"沟槽点{len(groove)}")
groove = groove[statistical_outlier_filter(groove)]

# ---------- 3. 裂缝融合测量 ----------
from src.fusion.late_fusion import measure_length_3d  # noqa: E402
length_m = measure_length_3d(groove)
err = abs(length_m - TRUE_LEN) / TRUE_LEN
check("裂缝长度误差<2%", err < 0.02, f"测得{length_m:.4f}m 真值{TRUE_LEN:.4f}m err={err:.3%}")

mpp = meters_per_pixel(5.0, K)
check("物理尺度", abs(mpp - 0.00625) < 1e-9, f"mpp={mpp}")
fused = measure_crack_fused_depth(4.0, 5.0, K, groove)
check("融合宽度", abs(fused["width_m"] - 0.025) < 1e-9, f"width_m={fused['width_m']}")

# ---------- 4. 剥落面积 ----------
uv_sp, _, _ = project_points(spall, K, R, t)
sx1, sy1 = uv_sp.min(axis=0) - 1
sx2, sy2 = uv_sp.max(axis=0) + 1
idx2 = points_in_mask(P, K, R, t, [[sx1, sy1], [sx2, sy1], [sx2, sy2], [sx1, sy2]], W, H)
area_m = measure_area_3d(P[idx2])
err_a = abs(area_m - TRUE_AREA) / TRUE_AREA
check("剥落面积误差<5%", err_a < 0.05, f"测得{area_m:.4f}sqm 真值{TRUE_AREA:.4f}sqm err={err_a:.3%}")

# ---------- 5. PLY 读写 ----------
ply_path = os.path.join(PROJECT_ROOT, "src", "fusion", "_selftest.ply")
small = P[:500]
with open(ply_path, "w") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(small)}\nproperty float x\nproperty float y\nproperty float z\n")
    f.write("end_header\n")
    np.savetxt(f, small, fmt="%.6f")
xyz_back, props = read_ply_ascii(ply_path)
check("PLY读写", np.allclose(xyz_back, small, atol=1e-6), f"props={props}")
os.remove(ply_path)

print()
if all(PASS):
    print(f"ALL {len(PASS)} CHECKS PASSED")
else:
    print(f"FAILED: {PASS.count(False)}/{len(PASS)}")
    sys.exit(1)
