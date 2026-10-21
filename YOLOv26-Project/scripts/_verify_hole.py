"""
_verify_hole.py — 生成 hole 复核接触表（12格/表 @500px）
候选来源（并集）：
  1. 我的标记（my_hole_flags.json，train 1-8 + val 1-3）
  2. 审查员报告（hole_recheck / hole_val_recheck / hole_test_recheck）
  3. 检测器 seam2 band >= 阈值（train>=12，val/test>=8）
输出：sheets_VH/v_XXX.jpg + manifest_VH.json（12格/表，5列×3行不足则调整）
"""
import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TMP = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\view_review")
AG = TMP / "agents"
ROOT = Path(r"D:\开始做项目\YOLOv26-Project\datasets\final_5class_clean")
EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def img_path(split, stem):
    for e in EXTS:
        p = ROOT / "images" / split / f"{stem}{e}"
        if p.exists():
            return p
    return None


def main():
    man = json.loads((TMP / "manifest_A.json").read_text(encoding="utf-8"))
    cand = {}

    def add(split, stem, src):
        cand.setdefault((split, stem), set()).add(src)

    # 1) 我的
    my = json.loads((TMP / "my_hole_flags.json").read_text(encoding="utf-8"))
    for sh, info in my.items():
        if sh.startswith("_"):
            continue
        for t in info.get("tiles", []):
            split, stem = man[sh][t]
            add(split, stem, "me")

    # 2) 审查员
    for fname, tag in [("hole_recheck.json", "rechk"),
                       ("hole_val_recheck.json", "valchk"),
                       ("hole_test_recheck.json", "tstchk")]:
        fp = AG / fname
        if not fp.exists():
            print(f"[WARN] missing {fname}")
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for sh, info in data.items():
            if not isinstance(info, dict) or info.get("clean"):
                continue
            if sh not in man:
                print(f"[WARN] sheet {sh} not in manifest")
                continue
            for fl in info.get("flags", []):
                t = fl.get("tile")
                if isinstance(t, int) and 0 <= t < len(man[sh]):
                    split, stem = man[sh][t]
                    add(split, stem, tag)

    # 3) 检测器
    dp = TMP / "seam2_hole.csv"
    if dp.exists():
        with open(dp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    band = float(r["band"])
                except (ValueError, KeyError):
                    continue
                thr = 12.0 if r["split"] == "train" else 8.0
                if band >= thr:
                    add(r["split"], r["stem"], f"det{r['band'].split('.')[0]}")

    rows = sorted(cand)
    print(f"[cand] {len(rows)} tiles")
    for s in ("train", "val", "test"):
        print(f"   {s}: {sum(1 for r in rows if r[0] == s)}")

    # 渲染
    outdir = TMP / "sheets_VH"
    outdir.mkdir(exist_ok=True)
    for f in outdir.glob("*.jpg"):
        f.unlink()
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    cols, tile = 4, 500
    per = 16
    vman = {}
    for si in range(math.ceil(len(rows) / per)):
        chunk = rows[si * per:(si + 1) * per]
        rows_n = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (tile * cols, tile * rows_n), (18, 18, 22))
        d = ImageDraw.Draw(sheet)
        for i, (split, stem) in enumerate(chunk):
            p = img_path(split, stem)
            im = None
            if p:
                try:
                    im = Image.open(p).convert("RGB")
                    im.thumbnail((tile - 8, tile - 8))
                except Exception:
                    im = None
            x, y = (i % cols) * tile, (i // cols) * tile
            if im:
                sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
            d.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(95, 95, 105))
            d.text((x + 8, y + 6), str(i), fill=(255, 235, 0), font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0))
        name = f"vh_{si + 1:03d}.jpg"
        sheet.save(outdir / name, quality=88)
        vman[name] = [[s, st] for s, st in chunk]
    (TMP / "manifest_VH.json").write_text(json.dumps(vman, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] sheets_VH: {len(vman)} sheets, manifest_VH.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
