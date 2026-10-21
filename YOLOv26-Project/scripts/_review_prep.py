"""
_review_prep.py — 数据审查准备（临时脚本）
==========================================
mode A: hole 图（全 split）生成 8x8 接触表（每格300px）+ 指标
mode B: 非 hole 图（全 split）生成 10x10 接触表（每格250px）+ 全量指标 + 分块清单

产出（临时目录）：
  sheets_A/hole_<split>_XXX.jpg    + manifest_A.json
  sheets_B/<split>_XXX.jpg         + manifest_B.json + chunks_B.json
  metrics.csv
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"D:\开始做项目\YOLOv26-Project\datasets\final_5class_clean")
OUT = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\view_review")
SPLITS = ["train", "val", "test"]
EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
NAMES = ["crack", "spalling", "hole", "rebar_exposed", "efflorescence"]


def get_font(size):
    for fp in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_classes(split):
    d = {}
    ldir = ROOT / "labels" / split
    if not ldir.exists():
        return d
    for lp in ldir.glob("*.txt"):
        txt = lp.read_text(encoding="utf-8", errors="ignore")
        cls = set()
        for line in txt.splitlines():
            p = line.split()
            if len(p) >= 5:
                try:
                    cls.add(int(float(p[0])))
                except ValueError:
                    pass
        d[lp.stem] = cls
    return d


def img_path(split, stem):
    for e in EXTS:
        p = ROOT / "images" / split / f"{stem}{e}"
        if p.exists():
            return p
    return None


def blur_score(im):
    try:
        import cv2
        g = np.asarray(im.convert("L").resize((256, 256)), dtype=np.uint8)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:
        return -1.0


def make_sheets(items, tile, cols, rows, outdir, prefix, manifest, metrics, font):
    per = cols * rows
    n = math.ceil(len(items) / per)
    for si in range(n):
        chunk = items[si * per:(si + 1) * per]
        sheet = Image.new("RGB", (tile * cols, tile * rows), (22, 22, 26))
        d = ImageDraw.Draw(sheet)
        for i, (split, stem, path) in enumerate(chunk):
            im = None
            try:
                im = Image.open(path).convert("RGB")
                metrics.append((split, stem, blur_score(im)))
                im.thumbnail((tile - 6, tile - 6))
            except Exception:
                im = None
            x = (i % cols) * tile
            y = (i // cols) * tile
            if im is not None:
                sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
            d.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(92, 92, 102))
            d.text((x + 7, y + 4), str(i), fill=(255, 235, 0), font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0))
        name = f"{prefix}_{si + 1:03d}.jpg"
        sheet.save(outdir / name, quality=86)
        manifest[name] = [[s, st] for s, st, _ in chunk]
        if (si + 1) % 20 == 0 or si == n - 1:
            print(f"  {prefix}: {si + 1}/{n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--chunk", type=int, default=28)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    metrics = []
    manifest = {}

    for split in SPLITS:
        cls_map = load_classes(split)
        stems = sorted(cls_map)
        if args.mode == "A":
            sel = [s for s in stems if 2 in cls_map[s]]
            what = "hole"
        else:
            sel = [s for s in stems if 2 not in cls_map[s]]
            what = "rest"
        items = []
        miss = 0
        for s in sel:
            p = img_path(split, s)
            if p is None:
                miss += 1
                continue
            items.append((split, s, p))
        print(f"[{split}] {what}: {len(items)} imgs (missing {miss})", flush=True)
        if args.mode == "A":
            outdir = OUT / "sheets_A"
            outdir.mkdir(exist_ok=True)
            make_sheets(items, 300, 8, 8, outdir, f"hole_{split}", manifest, metrics, get_font(30))
        else:
            outdir = OUT / "sheets_B"
            outdir.mkdir(exist_ok=True)
            make_sheets(items, 250, 10, 10, outdir, split, manifest, metrics, get_font(26))

    tag = "A" if args.mode == "A" else "B"
    (OUT / f"manifest_{tag}.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with open(OUT / f"metrics_{tag}.csv", "w", encoding="utf-8") as f:
        f.write("split,stem,blur\n")
        for s, st, b in metrics:
            f.write(f"{s},{st},{b:.2f}\n")

    if args.mode == "B":
        sheets = sorted(manifest)
        groups = [sheets[i:i + args.chunk] for i in range(0, len(sheets), args.chunk)]
        chunks = [{"chunk": f"chunk_{i + 1:02d}", "sheets": g} for i, g in enumerate(groups)]
        (OUT / "chunks_B.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[OK] mode B: {len(sheets)} sheets, {len(chunks)} chunks")

    print(f"[OK] mode {tag}: {len(manifest)} sheets, metrics {len(metrics)}")


if __name__ == "__main__":
    main()
