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


def _basis(cls_name, m, sev):
    """等级依据说明（供前端 tooltip 与报告展示）"""
    if cls_name == "crack":
        v, u, metric, bands = m.get("width_px", 0), "px", "裂缝宽度", [2, 5, 10, 20]
    elif cls_name == "hole":
        v, u, metric, bands = m.get("diameter_px", 0), "px", "等效直径", [30, 60, 120, 240]
    else:
        v, u, metric, bands = round(m.get("area_frac", 0) * 100, 2), "%", "面积占比", [3, 10, 20, 30]
    lo = bands[sev - 2] if sev > 1 else 0
    hi = bands[sev - 1] if sev <= 4 else "∞"
    return {"metric": metric, "value": v, "unit": u,
            "band": f"[{lo},{hi})",
            "short": f"{metric}{v}{u}"}


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
        b = _basis(cls_name, measurements, sev)
        return {"grade": sev, "name": gd["name"], "action": gd["action"],
                "needs_calibration": True, "basis": b,
                "basis_text": f"{b['short']}，属第{sev}档{b['band']}{b['unit']}；建议：{gd['action']}（像素相对分档，未做米制标定）"}

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
