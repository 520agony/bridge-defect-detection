"""生成 5 类接触表（每类抽样20张）用于人工判断拍摄视角（近景 vs 航拍）"""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"D:\开始做项目\YOLOv26-Project\datasets\final_5class_clean")
OUT = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\view5class")
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["crack", "spalling", "hole", "rebar_exposed", "efflorescence"]

TILE, COLS, N = 288, 4, 20
label_dir = ROOT / "labels" / "train"
img_dir = ROOT / "images" / "train"

try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()

# 收集每类的图stem
by_class = {i: [] for i in range(5)}
for lp in sorted(label_dir.glob("*.txt")):
    txt = lp.read_text(encoding="utf-8", errors="ignore")
    if not txt.strip():
        continue
    cids = {int(l.split()[0]) for l in txt.splitlines() if len(l.split()) >= 7}
    for c in cids:
        if c in by_class:
            by_class[c].append(lp.stem)

for cid, cname in enumerate(NAMES):
    stems = by_class[cid]
    step = max(1, len(stems) // N)
    picked = stems[::step][:N]
    rows = (len(picked) + COLS - 1) // COLS
    sheet = Image.new("RGB", (TILE * COLS, TILE * rows), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for i, stem in enumerate(picked):
        imgp = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                imgp = p
                break
        if imgp is None:
            continue
        try:
            im = Image.open(imgp).convert("RGB")
        except Exception:
            continue
        im.thumbnail((TILE - 8, TILE - 8))
        x = (i % COLS) * TILE
        y = (i // COLS) * TILE
        sheet.paste(im, (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2))
        draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=(80, 80, 90))
        draw.text((x + 6, y + 4), f"#{i+1}", fill=(255, 240, 0), font=font,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    out = OUT / f"train_{cid}_{cname}.jpg"
    sheet.save(out, quality=88)
    print(f"[OK] {cname}: {len(stems)} imgs, sampled {len(picked)} -> {out}")

# 全局随机抽样一张总览表
all_stems = sorted({s for c in by_class.values() for s in c})
rng = random.Random(42)
picked = rng.sample(all_stems, 20)
rows = (len(picked) + COLS - 1) // COLS
sheet = Image.new("RGB", (TILE * COLS, TILE * rows), (24, 24, 28))
draw = ImageDraw.Draw(sheet)
for i, stem in enumerate(picked):
    imgp = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            imgp = p
            break
    if imgp is None:
        continue
    im = Image.open(imgp).convert("RGB")
    im.thumbnail((TILE - 8, TILE - 8))
    x = (i % COLS) * TILE
    y = (i // COLS) * TILE
    sheet.paste(im, (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2))
    draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=(80, 80, 90))
    draw.text((x + 6, y + 4), f"#{i+1}", fill=(255, 240, 0), font=font,
              stroke_width=2, stroke_fill=(0, 0, 0))
sheet.save(OUT / f"train_overview_random.jpg", quality=88)
print(f"[OK] overview 20 -> {OUT / 'train_overview_random.jpg'}")
