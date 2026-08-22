"""
病害等级评估（7 类 + 像素/毫米标定）
====================================
- 有 px_per_mm：像素 → 毫米/面积比 → 用 evaluate_defect_grade 出正式等级（1~5）。
- 无 px_per_mm：给像素级「相对严重度」（1~5 档），标记 needs_calibration=True。
"""

from src.models.yolov26_improvements import BRIDGE_DEFECT_GRADES, evaluate_defect_grade


def _relative_severity(cls_name, m):
    """无标定时按像素量级给 1~5 档相对严重度（工程经验分档，可调）"""
    if cls_name == "crack":
        w = m.get("width_px", 0)
        if w < 2:
            return 1
        if w < 5:
            return 2
        if w < 10:
            return 3
        if w < 20:
            return 4
        return 5
    if cls_name == "hole":
        d = m.get("diameter_px", 0)
        if d < 30:
            return 1
        if d < 60:
            return 2
        if d < 120:
            return 3
        if d < 240:
            return 4
        return 5
    # 面积类（spalling/honeycomb/seepage/corrosion/rebar_exposed）
    f = m.get("area_frac", 0)
    if f < 0.03:
        return 1
    if f < 0.10:
        return 2
    if f < 0.20:
        return 3
    if f < 0.30:
        return 4
    return 5


def grade_defect(cls_name, measurements, px_per_mm=None):
    """对单个病害给出等级评估结果

    Args:
        cls_name: 病害类别名（7 类之一）
        measurements: measure_defect 返回的像素测量 dict
        px_per_mm: 像素/毫米比例；None 表示未标定

    Returns:
        dict: {"grade", "name", "action", "needs_calibration", "measurements_mm"?}
    """
    if cls_name not in BRIDGE_DEFECT_GRADES:
        return {"grade": 0, "name": "未知", "action": "人工判定", "needs_calibration": False}

    calibrated = px_per_mm is not None and px_per_mm > 0

    if not calibrated:
        sev = _relative_severity(cls_name, measurements)
        gd = BRIDGE_DEFECT_GRADES[cls_name]["grades"][sev]
        return {"grade": sev, "name": gd["name"], "action": gd["action"], "needs_calibration": True}

    # 已标定：像素 → 毫米 / 面积比
    meas_mm = {}
    if cls_name == "crack":
        meas_mm["width_mm"] = measurements.get("width_px", 0) / px_per_mm
        meas_mm["length_mm"] = measurements.get("length_px", 0) / px_per_mm
    elif cls_name == "hole":
        meas_mm["diameter_mm"] = measurements.get("diameter_px", 0) / px_per_mm
    elif cls_name == "rebar_exposed":
        meas_mm["rust_pct"] = measurements.get("area_frac", 0) * 100.0
    else:  # spalling / honeycomb / seepage / corrosion
        meas_mm["area_pct"] = measurements.get("area_frac", 0) * 100.0

    result = evaluate_defect_grade(cls_name, meas_mm)
    result["needs_calibration"] = False
    result["measurements_mm"] = {k: round(v, 3) for k, v in meas_mm.items()}
    return result
