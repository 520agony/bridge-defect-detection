"""
把 final_7class 图片统一 resize 到「最长边 ≤ 960」（只缩不放，保持宽高比）
==============================================================
- 只处理图片，标签是归一化坐标，无需改动
- 仅当 max(w,h) > 960 时才缩小，已 ≤960 的图原样跳过
- 多进程加速，进度打印

用法:  ./.venv/Scripts/python.exe scripts/resize_dataset.py
"""

import sys
from pathlib import Path
from PIL import Image, ImageOps
from multiprocessing import Pool

BASE = Path(__file__).resolve().parent.parent
DST = BASE / "datasets" / "final_7class"
MAX_SIDE = 960
JPEG_QUALITY = 95


def resize_one(img_path_str):
    img_path = Path(img_path_str)
    try:
        im = Image.open(img_path)
        im = ImageOps.exif_transpose(im)  # 修正 EXIF 旋转
        w, h = im.size
        max_side = max(w, h)
        if max_side <= MAX_SIDE:
            return ("skip", None)
        scale = MAX_SIDE / max_side
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        im = im.resize((new_w, new_h), Image.LANCZOS)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        # 按原扩展名保存
        fmt = img_path.suffix.lower()
        if fmt in (".jpg", ".jpeg"):
            im.save(img_path, "JPEG", quality=JPEG_QUALITY)
        elif fmt == ".png":
            im.save(img_path, "PNG")
        else:
            im.save(img_path)
        return ("resized", f"{w}x{h} -> {new_w}x{new_h}")
    except Exception as e:
        return ("error", f"{img_path.name}: {e}")


def main():
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    imgs = []
    for split in ("train", "val", "test"):
        d = DST / "images" / split
        for ext in exts:
            imgs.extend(d.glob(f"*{ext}"))
    imgs = [str(p) for p in imgs]
    total = len(imgs)
    print(f"共 {total} 张图片, 目标最长边 {MAX_SIDE}")

    n_resized = n_skip = n_err = 0
    with Pool(8) as pool:
        for i, (status, info) in enumerate(pool.imap(resize_one, imgs, chunksize=100), 1):
            if status == "resized":
                n_resized += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                print(f"[ERR] {info}")
            if i % 5000 == 0 or i == total:
                print(f"进度 {i}/{total}  已缩 {n_resized}  跳过 {n_skip}  出错 {n_err}", flush=True)

    print(f"\n完成: 缩小 {n_resized} 张, 跳过(已≤960) {n_skip} 张, 出错 {n_err} 张")


if __name__ == "__main__":
    main()
