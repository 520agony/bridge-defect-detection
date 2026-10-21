"""
结果级后融合：2D 检测 + 点云 -> 米制量化
==========================================
链路：YOLO 2D 框/mask -> 投影筛点 -> 去噪 -> 拟合平面 -> 算长度/面积（米）

约定（与 Blender/TLSynth 模板脚本配套）：
- 世界坐标单位：米
- 相机模型：X_cam = R @ X_world + t，uv = K @ X_cam / z
- meta.json 内含 K(3x3)、R(3x3)、t(3,)、W、H

依赖：numpy / scipy / cv2（项目 venv 已有，无需 open3d）
"""

import json

import cv2
import numpy as np
from scipy.spatial import ConvexHull, cKDTree


# ============================================================
# 相机与投影
# ============================================================
def load_camera(meta_path):
    """读 meta.json，返回 K, R, t, W, H"""
    with open(meta_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    K = np.asarray(m["K"], dtype=np.float64)
    R = np.asarray(m["R"], dtype=np.float64)
    t = np.asarray(m["t"], dtype=np.float64)
    return K, R, t, int(m["W"]), int(m["H"])


def project_points(P, K, R, t):
    """世界点(N,3) -> 像素uv(N,2)、深度(N,)、有效掩膜（剔除相机后方点）"""
    Xc = (R @ P.T).T + t
    z = Xc[:, 2]
    valid = z > 1e-6
    uv = np.zeros((P.shape[0], 2))
    uv[valid] = (K @ Xc[valid].T).T[:, :2] / z[valid, None]
    return uv, z, valid


def points_in_box(P, K, R, t, bbox, W, H):
    """投影落在框内（含图像边界裁剪）的点索引"""
    x1, y1, x2, y2 = bbox
    uv, _, valid = project_points(P, K, R, t)
    inside = (
        valid
        & (uv[:, 0] >= max(0, x1)) & (uv[:, 0] < min(W, x2))
        & (uv[:, 1] >= max(0, y1)) & (uv[:, 1] < min(H, y2))
    )
    return np.nonzero(inside)[0]


def points_in_mask(P, K, R, t, mask_poly, W, H):
    """投影落在 seg 多边形内的点索引（比 bbox 更准，优先用这个）"""
    poly = np.asarray(mask_poly, dtype=np.float32).reshape(-1, 1, 2)
    x1, y1 = poly[:, 0, 0].min(), poly[:, 0, 1].min()
    x2, y2 = poly[:, 0, 0].max(), poly[:, 0, 1].max()
    cand = points_in_box(P, K, R, t, (x1, y1, x2, y2), W, H)
    if len(cand) == 0:
        return cand
    uv, _, _ = project_points(P[cand], K, R, t)
    keep = np.array([cv2.pointPolygonTest(poly, (float(u), float(v)), False) >= 0
                     for u, v in uv])
    return cand[keep]


# ============================================================
# 点云清洗
# ============================================================
def statistical_outlier_filter(P, k=8, std_ratio=2.0):
    """numpy 版统计滤波：去离群噪声点，返回保留索引"""
    if len(P) <= k:
        return np.arange(len(P))
    tree = cKDTree(P)
    dists, _ = tree.query(P, k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    mu, sigma = mean_d.mean(), mean_d.std()
    return np.nonzero(mean_d <= mu + std_ratio * sigma)[0]


def fit_plane(P):
    """SVD 拟合平面，返回 (法向量n(3,), 中心c(3,), 平面基u,v)"""
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    return n, c, vt[0], vt[1]


# ============================================================
# 米制测量
# ============================================================
def measure_length_3d(P):
    """PCA 主方向极差 = 真实长度（米），用于裂缝"""
    if len(P) < 2:
        return 0.0
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    proj = (P - c) @ vt[0]
    return float(proj.max() - proj.min())


def measure_area_3d(P):
    """拟合平面 -> 投到2D -> 凸包面积（平方米），用于剥落/孔洞/泛碱"""
    if len(P) < 3:
        return 0.0
    n, c, u, v = fit_plane(P)
    q = np.stack([(P - c) @ u, (P - c) @ v], axis=1)
    try:
        hull = ConvexHull(q)
    except Exception:
        return 0.0
    return float(hull.volume)  # 2D 凸包的 volume 即面积


def meters_per_pixel(depth_m, K):
    """针孔模型：物理尺度 = 深度 / 焦距（米/像素）"""
    return float(depth_m) / float(K[0, 0])


def measure_crack_fused_depth(width_px, depth_median, K, P_crack):
    """裂缝：宽_m = 宽_px × mpp；长_m = 点云PCA直接量"""
    mpp = meters_per_pixel(depth_median, K)
    return {
        "width_m": round(width_px * mpp, 4),
        "length_m": round(measure_length_3d(P_crack), 4),
        "mpp": round(mpp, 6),
        "depth_median": round(float(depth_median), 3),
    }


def median_depth(P, K, R, t):
    """点集在相机系下的中位深度"""
    Xc = (R @ P.T).T + t
    z = Xc[:, 2]
    z = z[z > 1e-6]
    return float(np.median(z)) if len(z) else 0.0


# ============================================================
# PLY 读取（最小 ascii 解析，无需 open3d；TLSynth 导出即此格式）
# ============================================================
def read_ply_ascii(path):
    """读 ascii PLY，返回 (N,3) xyz 和属性名列表"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    assert lines[0].strip() == "ply", "not a ply file"
    n_vert, props, header_end = 0, [], 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("element vertex"):
            n_vert = int(s.split()[-1])
        elif s.startswith("property"):
            props.append(s.split()[-1])
        elif s == "end_header":
            header_end = i + 1
            break
    data = np.loadtxt(lines[header_end:header_end + n_vert])
    cols = {name: j for j, name in enumerate(props)}
    xyz = data[:, [cols["x"], cols["y"], cols["z"]]]
    return xyz, props
