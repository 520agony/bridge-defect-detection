"""
_seam_detect.py — 拼接缝/拼图候选检测器
========================================
signals:
  seamA : 全跨度强边缘覆盖率（硬边拼接，无留缝）
  gutterH/V : 均匀"留缝"条带（白/黑/灰分隔条）两侧对比
  stepH/V  : 平坦-平坦硬台阶（panel 直接对接）

usage:
  python _seam_detect.py calibrate            # 用我人工标注的 sheet1-3 校准
  python _seam_detect.py run                  # 全数据扫描 -> detect_all.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"D:\开始做项目\YOLOv26-Project\datasets\final_5class_clean")
TMP = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\view_review")
EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def detect_features(path):
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return None
    g = np.asarray(im.resize((512, 512)), dtype=np.float32)
    H, W = g.shape
    feats = {}

    # --- seamA: full-span strong edge coverage ---
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    Mw, Mh = int(W * 0.08), int(H * 0.08)
    col_cov = (dx > 32).mean(axis=0)
    row_cov = (dy > 32).mean(axis=1)
    c_max = float(col_cov[Mw:W - Mw].max()) if W - 2 * Mw > 2 else 0.0
    r_max = float(row_cov[Mh:H - Mh].max()) if H - 2 * Mh > 2 else 0.0
    feats["seamA"] = max(c_max, r_max)

    def gutter_and_step(gray, axis):
        """axis=1 -> 按行(水平条带); axis=0 -> 按列"""
        if axis == 1:
            std, mean, n = gray.std(axis=1), gray.mean(axis=1), gray.shape[0]
        else:
            std, mean, n = gray.std(axis=0), gray.mean(axis=0), gray.shape[1]
        m = max(3, int(n * 0.08))
        gbest = sbest = 0.0
        i = m
        while i < n - m:
            if std[i] < 7.0:
                j = i
                while j < n - m and std[j] < 7.0 and (j - i) < 16:
                    j += 1
                if j - m > i - 0 and i - 6 >= 0 and j + 6 <= n:
                    L = mean[max(0, i - 6):i].mean()
                    R = mean[j:j + 6].mean()
                    band = mean[i:j].mean()
                    c = max(abs(band - L), abs(band - R))
                    if j - i >= 1:
                        gbest = max(gbest, float(c))
                i = j
            else:
                i += 1
        # step: flat-flat sharp transition
        for k in range(m, n - m):
            d = abs(float(mean[k]) - float(mean[k + 1]))
            if d > 16 and std[k] < 10 and std[k + 1] < 10:
                sbest = max(sbest, d)
        return gbest, sbest

    gH, sH = gutter_and_step(g, 1)
    gV, sV = gutter_and_step(g, 0)
    feats["gutter"] = max(gH, gV)
    feats["step"] = max(sH, sV)
    feats["score"] = 2.2 * min(feats["seamA"], 1.0) * 100 + 0.5 * feats["gutter"] + 0.6 * feats["step"]
    return feats


def stem_path(split, stem):
    for e in EXTS:
        p = ROOT / "images" / split / f"{stem}{e}"
        if p.exists():
            return p
    return None


CALIB = {
    # sheet -> (collage tiles, unsure tiles)
    "hole_train_001.jpg": ([5, 6, 11, 13, 17, 19, 22, 23, 26, 27, 30, 36, 38, 40, 41, 44, 48, 50, 52, 57, 63], [4]),
    "hole_train_002.jpg": ([3, 4, 8, 10, 14, 15, 19, 20, 23, 25, 26, 29, 31, 35, 36, 38, 42, 44, 48, 50, 52, 54, 59, 63], [34]),
    "hole_train_003.jpg": ([1, 5, 6, 8, 10, 14, 16, 18, 20, 24, 25, 29, 36, 38, 41, 48, 50, 58, 59, 61, 62], [54]),
}


def calibrate():
    man = json.loads((TMP / "manifest_A.json").read_text(encoding="utf-8"))
    rows = []
    for sheet, (flags, unsure) in CALIB.items():
        stems = man[sheet]
        for idx, (split, stem) in enumerate(stems):
            p = stem_path(split, stem)
            f = detect_features(p) if p else None
            if f is None:
                continue
            lab = "COLLAGE" if idx in flags else ("unsure" if idx in unsure else "single")
            rows.append((f["score"], f["seamA"], f["gutter"], f["step"], lab, f"{sheet}#{idx}"))
    rows.sort(reverse=True)
    print(f"{'score':>7} {'seamA':>6} {'gutter':>7} {'step':>6}  label    ref")
    for sc, sa, gu, st, lab, ref in rows[:80]:
        print(f"{sc:7.1f} {sa:6.2f} {gu:7.1f} {st:6.1f}  {lab:8} {ref}")
    print("...")
    # 分类统计
    for lab in ["COLLAGE", "single", "unsure"]:
        vals = [r[0] for r in rows if r[4] == lab]
        if vals:
            vals.sort()
            print(f"{lab}: n={len(vals)} min={vals[0]:.1f} p25={vals[len(vals)//4]:.1f} "
                  f"med={vals[len(vals)//2]:.1f} p75={vals[3*len(vals)//4]:.1f} max={vals[-1]:.1f}")


def run_all():
    out = TMP / "detect_all.csv"
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "stem", "score", "seamA", "gutter", "step"])
        for split in ["train", "val", "test"]:
            imgs = sorted((ROOT / "images" / split).iterdir())
            for i, p in enumerate(imgs):
                if p.suffix.lower() not in EXTS:
                    continue
                feats = detect_features(p)
                if feats is None:
                    continue
                w.writerow([split, p.stem, f"{feats['score']:.1f}", f"{feats['seamA']:.3f}",
                            f"{feats['gutter']:.1f}", f"{feats['step']:.1f}"])
                n += 1
                if n % 2000 == 0:
                    print(f"  {n}", flush=True)
    print(f"[OK] {n} -> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    mode = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if mode == "calibrate":
        calibrate()
    else:
        run_all()
