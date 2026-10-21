r"""
_newdata_sheets.py — 6个新数据zip全量接触表（10x10/表）+ 清单
================================================================
输出: TEMP/newdata/sheets_ND/nd_XXX.jpg
      manifest_ND.json {sheet: [[zipidx, innername], ...]}
      metrics_ND.csv (split=pack, stem, blur)
"""
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIR = Path(r"D:\开始做项目\当前所需要的数据")
OUT = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\newdata")
SHEETS = OUT / "sheets_ND"
SHEETS.mkdir(parents=True, exist_ok=True)

ZIPS = [
    "Concrete.v1i.coco-segmentation.zip",
    "Efflorescence.v1i.coco-segmentation.zip",
    "Pothole_Segmentation_YOLOv8.v1i.coco-segmentation.zip",
    "Spalling and exposed rebar.v1i.coco-segmentation.zip",
    "pothole project.v1i.coco-segmentation.zip",
    "Potholes and Roads Instance Segmentation.v1i.yolov8.zip",
]

IMGSFX = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_font(size):
    for fp in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def blur_score(im):
    try:
        import cv2
        g = np.asarray(im.convert("L").resize((256, 256)), dtype=np.uint8)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        return -1.0


def main():
    # 收集全部 (zipidx, innername)
    items = []
    for zi, zn in enumerate(ZIPS):
        z = zipfile.ZipFile(DIR / zn)
        names = sorted(n for n in z.namelist() if n.lower().endswith(IMGSFX))
        print(f"[{zi}] {zn}: {len(names)} imgs", flush=True)
        for n in names:
            items.append((zi, n))
    print(f"total {len(items)}", flush=True)

    tile, cols, rows = 250, 10, 10
    per = cols * rows
    font = get_font(26)
    manifest = {}
    metrics = []
    nsheets = math.ceil(len(items) / per)
    # zip句柄缓存
    handles = {}

    def read_img(zi, n):
        if zi not in handles:
            handles[zi] = zipfile.ZipFile(DIR / ZIPS[zi])
        return handles[zi].read(n)

    for si in range(nsheets):
        chunk = items[si * per:(si + 1) * per]
        sheet = Image.new("RGB", (tile * cols, tile * rows), (22, 22, 26))
        d = ImageDraw.Draw(sheet)
        rec = []
        for i, (zi, n) in enumerate(chunk):
            try:
                im = Image.open(io.BytesIO(read_img(zi, n))).convert("RGB")
                metrics.append((zi, n, blur_score(im)))
                im.thumbnail((tile - 6, tile - 6))
            except Exception:
                im = None
                metrics.append((zi, n, -2.0))
            x, y = (i % cols) * tile, (i // cols) * tile
            if im is not None:
                sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
            d.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(92, 92, 102))
            d.text((x + 7, y + 4), str(i), fill=(255, 235, 0), font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0))
            rec.append([zi, n])
        name = f"nd_{si + 1:03d}.jpg"
        sheet.save(SHEETS / name, quality=86)
        manifest[name] = rec
        if (si + 1) % 10 == 0 or si == nsheets - 1:
            print(f"  sheet {si + 1}/{nsheets}", flush=True)

    (OUT / "manifest_ND.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "metrics_ND.csv", "w", encoding="utf-8") as f:
        f.write("zipidx,inner,blur\n")
        for zi, n, b in metrics:
            f.write(f"{zi},\"{n}\",{b:.1f}\n")
    # 分块清单（每chunk 14表）
    sheets = sorted(manifest)
    chunks = [sheets[i:i + 14] for i in range(0, len(sheets), 14)]
    ch = [{"chunk": f"ndc_{i + 1:02d}", "sheets": g} for i, g in enumerate(chunks)]
    (OUT / "chunks_ND.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {len(manifest)} sheets, {len(ch)} chunks")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
