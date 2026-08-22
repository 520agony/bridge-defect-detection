"""
评估报告生成
============
把检测 + 测量 + 等级结果组织成结构化 dict，并渲染为中文 Markdown 报告。
"""

from pathlib import Path

from src.analysis import CLASS_ZH


def build_report(image_path, image_size, defects, px_per_mm=None):
    """组织成结构化报告 dict

    Args:
        image_path: 图片路径
        image_size: (W, H)
        defects: list，每项含 class/class_id/confidence/bbox/measurements/grade
        px_per_mm: 标定比例（None = 未标定）
    """
    total = len(defects)
    max_grade = max((d["grade"]["grade"] for d in defects), default=0)
    return {
        "image": str(image_path),
        "image_size": image_size,
        "px_per_mm": px_per_mm,
        "calibrated": px_per_mm is not None,
        "total_defects": total,
        "max_grade": max_grade,
        "defects": defects,
    }


def _fmt_measurements(cls_name, m, m_mm=None):
    parts = []
    if "width_px" in m:
        parts.append(f"宽≈{m['width_px']}px")
    if "length_px" in m:
        parts.append(f"长≈{m['length_px']}px")
    if "diameter_px" in m:
        parts.append(f"直径≈{m['diameter_px']}px")
    if cls_name not in ("crack", "hole") and "area_px" in m:
        parts.append(f"面积≈{m['area_px']:.0f}px²")
    if cls_name not in ("crack", "hole") and "area_frac" in m:
        parts.append(f"占图≈{m['area_frac'] * 100:.2f}%")
    if m_mm:
        if "width_mm" in m_mm:
            parts.append(f"宽≈{m_mm['width_mm']:.2f}mm")
        if "length_mm" in m_mm:
            parts.append(f"长≈{m_mm['length_mm']:.1f}mm")
        if "diameter_mm" in m_mm:
            parts.append(f"直径≈{m_mm['diameter_mm']:.1f}mm")
        if "area_pct" in m_mm:
            parts.append(f"面积占比≈{m_mm['area_pct']:.1f}%")
        if "rust_pct" in m_mm:
            parts.append(f"锈蚀占比≈{m_mm['rust_pct']:.1f}%")
    if m.get("low_quality"):
        parts.append("⚠️低质量估算")
    return "，".join(parts) if parts else "—"


def render_markdown(report):
    lines = [
        "# 桥梁病害检测与评估报告",
        "",
        f"- **图片**: `{Path(report['image']).name}`",
        f"- **检测病害数**: {report['total_defects']}",
        f"- **最高病害等级**: {report['max_grade']} 级",
    ]
    if report["calibrated"]:
        lines.append(f"- **标定**: 已标定（{report['px_per_mm']} px/mm）")
    else:
        lines.append("- **标定**: ⚠️ 未标定（无参照物），等级为像素级相对严重度，仅供参考")
    lines += [
        "",
        "## 检测明细",
        "",
        "| # | 病害 | 置信度 | 测量值 | 等级 | 处置建议 |",
        "|---|------|--------|--------|------|----------|",
    ]
    for i, d in enumerate(report["defects"], 1):
        zh = d.get("zh") or CLASS_ZH.get(d["class"], d["class"])
        conf = f"{d['confidence'] * 100:.1f}%"
        meas = _fmt_measurements(d["class"], d["measurements"], d["grade"].get("measurements_mm"))
        g = d["grade"]
        gtext = f"{g['grade']}级·{g['name']}" + ("（需标定）" if g.get("needs_calibration") else "")
        lines.append(f"| {i} | {zh} | {conf} | {meas} | {gtext} | {g['action']} |")
    lines += [
        "",
        "## 说明",
        "",
        "- 测量基于检测框内图像处理（骨架化 + 距离变换）；模型仅输出 bbox（无分割 mask），低对比度裂缝误差较大。",
        "- 无参照物时无法确定像素→毫米比例，等级阈值（mm）无法直接套用，上表等级为像素级相对严重度。",
        "- 有参照物时请用 `--px-per-mm` 传入比例，即可输出正式毫米级等级。",
        "",
    ]
    return "\n".join(lines)
