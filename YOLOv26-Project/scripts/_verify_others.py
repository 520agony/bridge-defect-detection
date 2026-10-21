"""_verify_others.py — 非 hole 类复核表（排除重复类，重复走算法留一删一）"""
import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TMP = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\view_review")
ROOT = Path(r"D:\开始做项目\YOLOv26-Project\datasets\final_5class_clean")
EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def img_path(split, stem):
    for e in EXTS:
        p = ROOT / "images" / split / f"{stem}{e}"
        if p.exists():
            return p
    return None


def main():
    vh = set()
    mvh = TMP / "manifest_VH.json"
    if mvh.exists():
        m = json.loads(mvh.read_text(encoding="utf-8"))
        for arr in m.values():
            for s, st in arr:
                vh.add((s, st))
    rows = []
    with open(TMP / "candidates.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            reasons = r["reasons"]
            if reasons == "duplicate":
                continue  # 纯重复走算法
            if (r["split"], r["stem"]) in vh:
                continue  # 已在 hole 复核队列
            rows.append((r["split"], r["stem"], reasons))
    print(f"[cand-others] {len(rows)}")
    outdir = TMP / "sheets_VO"
    outdir.mkdir(exist_ok=True)
    for f in outdir.glob("*.jpg"):
        f.unlink()
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    cols, tile, per = 4, 500, 16
    vman = {}
    for si in range(math.ceil(len(rows) / per)):
        chunk = rows[si * per:(si + 1) * per]
        rn = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (tile * cols, tile * rn), (18, 18, 22))
        d = ImageDraw.Draw(sheet)
        for i, (split, stem, _) in enumerate(chunk):
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
        name = f"vo_{si + 1:03d}.jpg"
        sheet.save(outdir / name, quality=88)
        vman[name] = [list(t[:2]) for t in chunk]
    (TMP / "manifest_VO.json").write_text(json.dumps(vman, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] sheets_VO: {len(vman)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
