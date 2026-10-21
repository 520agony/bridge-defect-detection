"""
_seam2.py — 升级版拼缝检测（全跨度直缝 + 两侧内容差异）
=====================================================
对每张图：
  - 找"几乎贯穿整幅"的直线缝（列缝/行缝，剔除 8% 边距）
  - 缝两侧各取 22px 宽条带，比较均值差 dmean 与纹理差 dstd
  - band_score = coverage * (dmean + 0.6*dstd)
真拼图：coverage 高 且 两侧差异大；车道线/裂缝：两侧相近 -> 低分；路缘：可能中等。
输出 CSV：split,stem,cov,pos,dmean,dstd,band
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"D:\开始做项目\YOLOv26-Project\datasets\final_5class_clean")
OUT = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\view_review\seam2_hole.csv")
EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def analyze(path):
    im = Image.open(path).convert("L").resize((512, 512))
    g = np.asarray(im, dtype=np.float32)
    H, W = g.shape
    best = (0.0, 0, 0.0, 0.0, 0.0)  # band, cov, pos, dmean, dstd

    def scan(gray, axis):
        # axis=1: 列缝(竖向); axis=0: 行缝
        if axis == 1:
            d = np.abs(np.diff(gray, axis=1))  # H x (W-1)
            n = W - 1
            get_band = lambda j: (gray[:, max(0, j - 24):max(1, j - 2)],
                                  gray[:, min(n, j + 2):min(n, j + 24)])
        else:
            d = np.abs(np.diff(gray, axis=0))  # (H-1) x W
            n = H - 1
            get_band = lambda i: (gray[max(0, i - 24):max(1, i - 2), :],
                                  gray[min(n, i + 2):min(n, i + 24), :])
        cov = (d > 30).mean(axis=0 if axis == 1 else 1)  # per position
        m = int(n * 0.08)
        lo, hi = m, n - m
        if hi <= lo:
            return None
        cand = np.argsort(-cov[lo:hi])[:6] + lo
        out = (0.0, 0, 0.0, 0.0, 0.0)
        for p in cand:
            c = float(cov[p])
            lb, rb = get_band(int(p))
            if lb.size == 0 or rb.size == 0:
                continue
            dmean = float(abs(lb.mean() - rb.mean()))
            dstd = float(abs(lb.std() - rb.std()))
            band = c * (dmean + 0.6 * dstd)
            if band > out[0]:
                out = (band, c, int(p), dmean, dstd)
        return out

    for ax in (1, 0):
        r = scan(g, ax)
        if r and r[0] > best[0]:
            best = r
    return best


def main():
    rows = []
    n = 0
    for split in ("train", "val", "test"):
        d = ROOT / "images" / split
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in EXTS:
                continue
            # 只算 hole：labels 含 class 2
            lp = ROOT / "labels" / split / f"{p.stem}.txt"
            if not lp.exists():
                continue
            txt = lp.read_text(encoding="utf-8", errors="ignore")
            if not any(len(l.split()) >= 7 and l.split()[0] in ("2", "2.0") for l in txt.splitlines()):
                continue
            band, cov, pos, dmean, dstd = analyze(p)
            rows.append((split, p.stem, f"{band:.1f}", f"{cov:.2f}", pos, f"{dmean:.0f}", f"{dstd:.0f}"))
            n += 1
            if n % 300 == 0:
                print(n, flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "stem", "band", "cov", "pos", "dmean", "dstd"])
        w.writerows(rows)
    print(f"[OK] {n} -> {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
