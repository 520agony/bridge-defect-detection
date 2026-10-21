"""B3 Gradio演示页（2D版，权重可换）
================================
Tab1 单图检测：上传桥梁图 → YOLO-seg mask描边 + 像素测量 + 等级
Tab2 融合演示B2：sample02/03预设 → 框/mask投影筛点 → 米制 vs 真值

启动（项目根）：
    .\\.venv\\Scripts\\python.exe gradio_demo\\app.py
    浏览器开 http://127.0.0.1:7860
换权重：顶部文本框填新best.pt点“加载模型”（m/distill下来直接换，无需改代码）
"""
import json
import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
SAMPLE_ROOT = Path(r"D:\开始做项目\fusion_sim")
DEFAULT_WEIGHT = PROJ / "experiments" / "seg_v2_clean" / "weights" / "best.pt"

ZH = {"crack": "裂缝", "spalling": "剥落", "hole": "孔洞",
      "rebar_exposed": "露筋", "efflorescence": "泛碱"}
COLOR = {"crack": (25, 25, 230), "spalling": (49, 130, 245), "hole": (75, 180, 60),
         "rebar_exposed": (244, 212, 66), "efflorescence": (250, 139, 167)}  # RGB

_model, _model_path = None, ""


def load_model(weight_path):
    """加载/切换权重，返回状态文本"""
    global _model, _model_path
    from ultralytics import YOLO
    p = Path(weight_path)
    if not p.exists():
        return f"❌ 权重不存在: {p}"
    _model = YOLO(str(p))
    _model_path = str(p)
    return f"✅ 已加载: {p.name}（{p.stat().st_size/1e6:.1f}MB） 类别={list(_model.names.values())}"


def _predict(bgr, conf, imgsz):
    from src.analysis.measure import measure_defect
    from src.analysis.grade import grade_defect
    if _model is None:
        raise RuntimeError("模型未加载，先点“加载模型”")
    h, w = bgr.shape[:2]
    r = next(iter(_model.predict(bgr, conf=conf, imgsz=imgsz, verbose=False)))
    dets = []
    nb = 0 if r.boxes is None else len(r.boxes)
    for i in range(nb):
        b = r.boxes[i]
        cid = int(b.cls[0])
        cname = r.names[cid]
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        d = {"class": cname, "zh": ZH.get(cname, cname),
             "confidence": round(float(b.conf[0]), 4),
             "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]}
        if r.masks is not None:
            try:
                poly_n = r.masks.xyn[i].reshape(-1, 2)
                d["mask"] = [[round(float(u) * w, 1), round(float(v) * h, 1)]
                             for u, v in poly_n]
            except Exception:
                pass
        try:
            d["measurements"] = measure_defect(cname, bgr, d["bbox"], mask=d.get("mask"))
            d["grade"] = grade_defect(cname, d["measurements"])
        except Exception as e:
            d["measurements"], d["grade"] = {}, {"grade": 0, "name": f"测量失败:{e}"}
        dets.append(d)
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    ms = round(float(r.speed.get("inference", 0)), 2)
    return dets, ms


_FONT = None


def _font():
    """中文字体（解决cv2.putText画中文变????）"""
    global _FONT
    if _FONT is None:
        from PIL import ImageFont
        for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
            if Path(p).exists():
                _FONT = ImageFont.truetype(p, 22)
                break
        else:
            _FONT = ImageFont.load_default()
    return _FONT


def _put_zh(bgr, text, org, color_bgr):
    """PIL画中文标签，失败回退英文"""
    try:
        from PIL import Image, ImageDraw
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        dr = ImageDraw.Draw(pil)
        x, y = org
        tw, th = dr.textbbox((0, 0), text, font=_font())[2:]
        y0 = max(0, y - th - 6)
        dr.rectangle([x, y0, x + tw + 8, y0 + th + 6],
                     fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
        dr.text((x + 4, y0 + 3), text, font=_font(), fill=(10, 13, 18))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(bgr, text.encode("ascii", "replace").decode(),
                    org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
        return bgr


def _annotate(bgr, dets):
    vis = bgr.copy()
    for d in dets:
        c = COLOR.get(d["class"], (0, 255, 0))
        cbgr = (c[2], c[1], c[0])
        if d.get("mask"):
            pts = np.array(d["mask"], dtype=np.int32).reshape(-1, 1, 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], cbgr)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
            cv2.polylines(vis, [pts], True, cbgr, 2)
        else:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), cbgr, 2)
        x1, y1 = int(d["bbox"][0]), int(d["bbox"][1])
        vis = _put_zh(vis, f"{d['zh']} {d['confidence']:.2f}", (x1, max(0, y1 - 8)), cbgr)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def tab_detect(img_rgb, conf, imgsz):
    if img_rgb is None:
        return None, {}, "先上传图片"
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    dets, ms = _predict(bgr, conf, int(imgsz))
    md = f"检出 **{len(dets)}** 处 · 推理 {ms} ms · `{Path(_model_path).name}`\n\n"
    for d in dets:
        g = d.get("grade", {})
        md += f"- **{d['zh']}** {d['confidence']:.2f}｜{g.get('grade','?')}级·{g.get('name','')}｜{json.dumps(d.get('measurements', {}), ensure_ascii=False)}\n"
    out = [{k: v for k, v in d.items() if k != "mask"} for d in dets]
    return _annotate(bgr, dets), out, md if dets else md + "未检出病害"


def tab_fusion(sample, conf):
    from src.fusion.late_fusion import (load_camera, read_ply_ascii, points_in_box,
        points_in_mask, statistical_outlier_filter, measure_length_3d,
        measure_area_3d, median_depth)
    sdir = SAMPLE_ROOT / sample
    rgb_p, meta_p, ply_p, truth_p = (sdir / "rgb.png", sdir / "meta.json",
        sdir / "cloud.ply", sdir / "truth.json")
    if not rgb_p.exists():
        return None, {}, f"{sdir} 还没渲（Blender跑blender_sim01.py）"
    bgr = cv2.imdecode(np.frombuffer(rgb_p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    K, R, t, W, H = load_camera(str(meta_p))
    P, _ = read_ply_ascii(str(ply_p))
    truth = json.loads(truth_p.read_text(encoding="utf-8"))
    dets, ms = _predict(bgr, conf, 960)
    rows = []
    for d in dets:
        idx = points_in_mask(P, K, R, t, d["mask"], W, H) if d.get("mask") \
            else points_in_box(P, K, R, t, d["bbox"], W, H)
        Pc = P[idx]
        keep = statistical_outlier_filter(Pc) if len(Pc) > 8 else np.arange(len(Pc))
        Pf = Pc[keep]
        row = {"类别": d["zh"], "conf": d["confidence"],
               "点数": f"{len(Pf)}/{len(idx)}"}
        if len(Pf) >= 3:
            row["深度m"] = round(float(median_depth(Pf, K, R, t)), 3)
            if d["class"] == "crack":
                L = float(measure_length_3d(Pf))
                row["融合"] = f"{L:.4f} m"
                row["真值"] = f"{truth['crack_length_m']} m"
                row["误差"] = f"{abs(L-truth['crack_length_m'])/truth['crack_length_m']*100:.2f}%"
            elif d["class"] == "spalling":
                a = float(measure_area_3d(Pf))
                row["融合"] = f"{a:.6f} ㎡"
                row["真值"] = f"{truth['spall_area_sqm']} ㎡"
                row["误差"] = f"{abs(a-truth['spall_area_sqm'])/truth['spall_area_sqm']*100:.2f}%"
            else:
                row["融合"] = f"{float(measure_area_3d(Pf)):.6f} ㎡"
                row["真值"], row["误差"] = "—", "—"
        else:
            row["融合"] = "点太少"
            row["真值"], row["误差"] = "—", "—"
        rows.append(row)
    md = f"真值：裂缝 {truth['crack_length_m']} m｜剥落 {truth['spall_area_sqm']} ㎡\n\n"
    md += "融合链：YOLO框/mask → 投影筛点 → 统计滤波 → PCA长/凸包面积\n\n"
    md += "纯图像只能给像素，融合后给米/㎡——这就是B2要讲的故事。" if rows else "该阈值下0检出，把conf调低（如0.05）再试。"
    return _annotate(bgr, dets), rows, md


with gr.Blocks(title="智巡飞眼·B3演示") as demo:
    gr.Markdown("# 智巡飞眼 — B3演示页（2D版）\nYOLO-seg桥梁病害检测 + 点云后融合米制（权重可换）")
    with gr.Row():
        wbox = gr.Textbox(value=os.environ.get("MODEL_PATH", str(DEFAULT_WEIGHT)),
                          label="权重路径", scale=4)
        wbtn = gr.Button("加载模型", scale=1)
    wstat = gr.Markdown("点“加载模型”开始")
    wbtn.click(load_model, wbox, wstat)

    with gr.Tab("单图检测"):
        with gr.Row():
            inp = gr.Image(type="numpy", label="上传桥梁图")
            out_img = gr.Image(label="检测结果（mask描边）")
        with gr.Row():
            conf = gr.Slider(0.05, 0.8, 0.25, step=0.05, label="置信度")
            imgsz = gr.Radio(["640", "960"], value="960", label="推理尺寸")
            btn = gr.Button("智能检测", variant="primary")
        out_json = gr.JSON(label="检出明细（含测量+等级）")
        out_md = gr.Markdown()
        btn.click(tab_detect, [inp, conf, imgsz], [out_img, out_json, out_md])

    with gr.Tab("融合演示B2"):
        with gr.Row():
            sample = gr.Dropdown(["sample03", "sample02", "sample01"],
                                 value="sample03", label="仿真场景")
            conf2 = gr.Slider(0.05, 0.8, 0.25, step=0.05, label="置信度")
            btn2 = gr.Button("跑融合", variant="primary")
        out_img2 = gr.Image(label="检出")
        out_json2 = gr.JSON(label="融合 vs 真值")
        out_md2 = gr.Markdown()
        btn2.click(tab_fusion, [sample, conf2], [out_img2, out_json2, out_md2])


if __name__ == "__main__":
    print(load_model(os.environ.get("MODEL_PATH", str(DEFAULT_WEIGHT))))
    demo.launch(server_name="127.0.0.1", server_port=7860)
