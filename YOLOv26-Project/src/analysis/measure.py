"""
病害量化测量模块（v2 - 用 mask 测量）
=====================================
裂缝：用 mask 骨架+距离变换测宽/长（比 bbox 精确得多）
其他：优先用 mask 像素数算真实面积；无 mask 时退回 bbox 估算
"""

import math

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology


def _mask_to_binary(mask_poly, H, W):
    """多边形 [[x,y],...] → 原图尺寸的二值 mask"""
    if not mask_poly or len(mask_poly) < 3:
        return None
    arr = np.array(mask_poly, dtype=np.int32).reshape(-1, 1, 2)
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [arr], 1)
    return m


def measure_crack(img_bgr, bbox, mask_poly=None, pad=8):
    """测裂缝宽/长：优先用 mask 骨架+距离变换（精确）；无 mask 时退回 bbox 内阈值法"""
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)
    bw, bh = x2 - x1, y2 - y1
    diag = math.hypot(bw, bh)

    def fallback(reason):
        fw = max(1.0, float(bbox[2]) - float(bbox[0]))
        fh = max(1.0, float(bbox[3]) - float(bbox[1]))
        return {
            "width_px": round(min(fw, fh), 2),
            "length_px": round(max(fw, fh), 1),
            "skeleton_px": 0,
            "estimated": True,
            "low_quality": True,
            "reason": reason,
        }

    if bw < 3 or bh < 3:
        return fallback("bbox_too_small")

    # 优先：用 mask
    if mask_poly and len(mask_poly) >= 3:
        mask_bin_full = _mask_to_binary(mask_poly, H, W)
        if mask_bin_full is not None and mask_bin_full.sum() >= 5:
            mask_bin = mask_bin_full[y1:y2, x1:x2] > 0
            if int(mask_bin.sum()) >= 5:
                return _crack_from_mask(mask_bin, diag, reason_fallback=fallback)

    # 回退：bbox 内阈值法
    crop = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask_bin = _extract_crack_mask(gray) > 0
    if int(mask_bin.sum()) < 5:
        return fallback("no_crack_pixels")
    return _crack_from_mask(mask_bin, diag, reason_fallback=fallback)


def _crack_from_mask(mask_bin, diag, reason_fallback):
    """用 mask 二值图测宽/长"""
    skel = morphology.skeletonize(mask_bin)
    skel_px = int(skel.sum())
    if skel_px < 3:
        return reason_fallback("skeleton_too_short")
    if skel_px > 3.0 * diag:
        return reason_fallback("skeleton_too_branched")

    dist = ndimage.distance_transform_edt(mask_bin)
    half = dist[skel]
    half = half[half > 0]
    if half.size == 0:
        half = np.array([0.5])

    return {
        "width_px": round(max(1.0, 2.0 * float(np.median(half)) - 1.0), 2),
        "width_max_px": round(max(1.0, 2.0 * float(half.max()) - 1.0), 2),
        "length_px": round(float(skel_px), 1),
        "skeleton_px": skel_px,
        "estimated": False,
        "low_quality": False,
    }


def _extract_crack_mask(gray):
    """bbox 内阈值法（fallback）"""
    h, w = gray.shape
    bs = min(21, h, w)
    if bs % 2 == 0:
        bs -= 1
    bs = max(3, bs)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 7
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return np.zeros_like(th)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def _mask_area_px(mask_poly, H, W):
    """多边形 mask 像素数"""
    if not mask_poly or len(mask_poly) < 3:
        return None
    arr = np.array(mask_poly, dtype=np.int32).reshape(-1, 1, 2)
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [arr], 1)
    return int(m.sum())


def _mask_diameter_px(mask_poly, H, W):
    """多边形 mask 等效直径（按面积推算 + 最小包围框）"""
    area_px = _mask_area_px(mask_poly, H, W)
    if area_px is None or area_px == 0:
        return None
    # 等效直径 = 2*sqrt(area/π)
    return round(2.0 * math.sqrt(area_px / math.pi), 2)


def measure_defect(cls_name, img_bgr, bbox, mask=None):
    """按病害类型做像素级测量，返回统一 dict

    crack       -> mask 骨架+距离变换（width_px / length_px）
    hole        -> mask 等效直径
    其余面积类  -> mask 像素数（真实面积）；无 mask 时退回 bbox 估算
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    bbox_area = bw * bh
    img_area = float(H * W)

    mask_area = _mask_area_px(mask, H, W) if mask else None

    if cls_name == "crack":
        m = measure_crack(img_bgr, bbox, mask_poly=mask)
    elif cls_name == "hole":
        if mask_area:
            m = {"diameter_px": round(2.0 * math.sqrt(mask_area / math.pi), 2),
                 "area_px": mask_area}
        else:
            m = {"diameter_px": round(min(bw, bh), 2), "area_px": round(bbox_area, 1)}
    else:
        # 面积类：mask 像素数 = 真实面积（比 bbox 准得多）
        if mask_area is not None:
            m = {"area_px": mask_area, "area_frac": round(mask_area / img_area, 6)}
        else:
            m = {"area_px": round(bbox_area, 1), "area_frac": round(bbox_area / img_area, 6)}

    m["bbox_w_px"] = round(bw, 1)
    m["bbox_h_px"] = round(bh, 1)
    return m
