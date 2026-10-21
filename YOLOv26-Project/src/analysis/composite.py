"""
复合病害提示（规则层，非模型检出）
==================================
场景：大面积剥落内嵌套露筋钢筋网，模型只报剥落（实测 sim/真图均漏检，
降阈值/裁剪放大均证伪，见 nest 诊断）。
做法：大剥落 mask 内做锈色线状结构检测 → 命中则挂 "rule_hint" 提示。
诚实性：调用方必须将其标记为规则提示（虚线框 + 徽章），不计入 mAP/统计。
"""
import cv2
import numpy as np

# 可调阈值（调参记录，2阳性 + 6阴性，2026-10-20）：
#   lib_spalling_2（阳） rust=0.1651 lineden=6.18   -> 命中
#   lib_spalling_3（阳） rust=0.1247 lineden=54.09  -> 命中
#   b20_105（阴/实含细钢筋棍） 0.4382/45.64 -> 命中（可接受）
#   b20_159（阴/深色糙面） 0.1565/5.02 -> 排除
#   b20_216（阴/骨料外露） 0.0520/41.30 -> 排除
#   其余3阴：双低 -> 排除
RUST_RATIO_MIN = 0.10   # mask 内锈色像素占比
LINE_DENSITY_MIN = 6.0  # 总直线长度 / sqrt(mask面积)
SPALL_FRAC_MIN = 0.08   # 触发规则的剥落占图比下限（后端 gate 用）


def _mask_in_crop(mask_poly, bbox, cw, ch):
    x1, y1, _, _ = [int(round(v)) for v in bbox]
    pts = np.array([(p[0] - x1, p[1] - y1) for p in mask_poly], dtype=np.int32)
    m = np.zeros((ch, cw), dtype=np.uint8)
    cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 1)
    return m


def suggest_rebar_in_spall(img_bgr, mask_poly, bbox):
    """返回 (hit, info)。info 含 rust_ratio / line_density / rust_bbox(原图坐标)。"""
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return False, {"reason": "too_small"}
    crop = img_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    m = _mask_in_crop(mask_poly, (x1, y1, x2, y2), cw, ch)
    mask_area = int(m.sum())
    if mask_area < 500:
        return False, {"reason": "mask_too_small"}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    rust = cv2.inRange(hsv, np.array([8, 70, 40]), np.array([25, 255, 220]))
    rust_in = cv2.bitwise_and(rust, rust, mask=m)
    rust_ratio = float(rust_in.sum() / 255) / mask_area

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=m)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=m)
    min_len = int(max(cw, ch) * 0.15)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=min_len, maxLineGap=8)
    total = 0.0
    if lines is not None:
        for x1l, y1l, x2l, y2l in lines[:, 0]:
            total += float(((x2l - x1l) ** 2 + (y2l - y1l) ** 2) ** 0.5)
    line_density = total / (mask_area ** 0.5)

    # 锈区外接框（原图坐标），用于提示框
    ys, xs = np.nonzero(rust_in)
    if len(xs) == 0:
        return False, {"rust_ratio": round(rust_ratio, 4), "line_density": round(line_density, 3)}
    rx1, rx2 = int(xs.min()) + x1, int(xs.max()) + x1
    ry1, ry2 = int(ys.min()) + y1, int(ys.max()) + y1
    info = {"rust_ratio": round(rust_ratio, 4), "line_density": round(line_density, 3),
            "rust_bbox": [rx1, ry1, rx2, ry2]}
    hit = rust_ratio >= RUST_RATIO_MIN and line_density >= LINE_DENSITY_MIN
    return hit, info
