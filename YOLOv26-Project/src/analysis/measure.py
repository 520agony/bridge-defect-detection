"""
病害量化测量模块
================
对检测到的病害 bbox 做像素级量化测量。

裂缝测量（核心）：bbox 内图像处理 —— 灰度化 → 提取裂缝像素 → 保留最大连通域 → 骨架化 → 距离变换
  - 宽度 = 2 × 距离变换值（骨架点到最近背景的距离，即半宽）
  - 长度 = 骨架像素数（近似）

其他病害：基于 bbox 面积/尺寸的代理测量。

限制：模型只出 bbox（无分割 mask），裂缝像素靠阈值提取；低对比度/纹理裂缝提取不可靠时，
      measure_crack 会回退到 bbox 估算并标记 low_quality=True（报告里会注明）。
"""

import math

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology


def _extract_crack_mask(gray):
    """提取裂缝像素（暗色细纹），保留最大连通域以抑制纹理噪声"""
    h, w = gray.shape
    # blockSize 需为奇数且 <= 图像尺寸
    bs = min(21, h, w)
    if bs % 2 == 0:
        bs -= 1
    bs = max(3, bs)

    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 7
    )

    # 保留最大连通域（主裂缝），去除散布纹理噪声。
    # 注意：不做闭运算/膨胀，否则会把裂缝加宽，导致宽度测量偏大。
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return np.zeros_like(th)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def measure_crack(img_bgr, bbox, pad=8):
    """测量裂缝的像素宽度与长度（骨架 + 距离变换）

    Args:
        img_bgr: 原始 BGR 图
        bbox: [x1, y1, x2, y2]（像素坐标）
        pad: bbox 外扩像素（保留裂缝边缘）

    Returns:
        dict: {"width_px", "width_max_px", "length_px", "skeleton_px",
               "estimated", "low_quality", "reason"?}
    """
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    H, W = img_bgr.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    cw, ch = x2 - x1, y2 - y1
    diag = math.hypot(cw, ch)

    def fallback(reason):
        """提取失败/不可靠时，退回 bbox 尺寸估算并标记低质量"""
        bw = max(1.0, float(bbox[2]) - float(bbox[0]))
        bh = max(1.0, float(bbox[3]) - float(bbox[1]))
        return {
            "width_px": round(min(bw, bh), 2),
            "length_px": round(max(bw, bh), 1),
            "skeleton_px": 0,
            "estimated": True,
            "low_quality": True,
            "reason": reason,
        }

    if cw < 3 or ch < 3:
        return fallback("bbox_too_small")

    crop = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 不做高斯模糊：模糊会把细裂缝（1~3px）加宽，导致宽度测量偏大
    # 降噪由后续「保留最大连通域」承担

    mask = _extract_crack_mask(gray)
    mask_bin = mask > 0
    if int(mask_bin.sum()) < 5:
        return fallback("no_crack_pixels")

    skel = morphology.skeletonize(mask_bin)
    skel_px = int(skel.sum())
    if skel_px < 3:
        return fallback("skeleton_too_short")

    # 质量检查：干净裂缝的骨架长度应接近其跨度（<= 对角线若干倍）；
    # 若远大于对角线，说明提取到的是分叉纹理而非单条裂缝
    if skel_px > 3.0 * diag:
        return fallback("skeleton_too_branched")

    # 距离变换：裂缝像素到最近背景的距离
    # 骨架位于裂缝中轴，宽度 = 2×距离 − 1（距离变换算到背景像素，含 +1 离轴偏移）
    dist = ndimage.distance_transform_edt(mask_bin)
    half = dist[skel]
    half = half[half > 0]
    if half.size == 0:
        half = np.array([0.5])

    width_px = max(1.0, 2.0 * float(np.median(half)) - 1.0)
    width_max_px = max(1.0, 2.0 * float(half.max()) - 1.0)

    return {
        "width_px": round(width_px, 2),
        "width_max_px": round(width_max_px, 2),
        "length_px": round(float(skel_px), 1),
        "skeleton_px": skel_px,
        "estimated": False,
        "low_quality": False,
    }


def measure_defect(cls_name, img_bgr, bbox):
    """按病害类型做像素级测量，返回统一 dict

    crack       -> 骨架+距离变换（width_px / length_px）
    hole        -> bbox 等效直径（diameter_px）
    其余面积类  -> bbox 面积 + 占图面积比（area_px / area_frac）
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    bbox_area = bw * bh
    img_area = float(H * W)

    if cls_name == "crack":
        m = measure_crack(img_bgr, bbox)
    elif cls_name == "hole":
        m = {"diameter_px": round(min(bw, bh), 2), "area_px": round(bbox_area, 1)}
    else:
        m = {"area_px": round(bbox_area, 1), "area_frac": round(bbox_area / img_area, 6)}

    m["bbox_w_px"] = round(bw, 1)
    m["bbox_h_px"] = round(bh, 1)
    return m
