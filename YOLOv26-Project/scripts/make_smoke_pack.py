"""
make_smoke_pack.py - Kaggle 冒烟小包（本机跑）
================================================
从 final_5class/train 按类分层抽 1000 张（每类至少 150 张含该类图），
生成 server_pack/smoke_data.zip + configs/data_smoke.yaml。
Kaggle 上只验证链路，不看指标。

跑法：.venv/Scripts/python.exe scripts/make_smoke_pack.py
"""
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "datasets" / "final_5class" / "images" / "train"
LAB = BASE / "datasets" / "final_5class" / "labels" / "train"
OUT = BASE / "server_pack" / "smoke_data"
N_PER_CLASS, SEED = 200, 0

NC = 5


def main():
    random.seed(SEED)
    by_cls = defaultdict(list)
    for f in os.listdir(LAB):
        if not f.endswith(".txt"):
            continue
        seen = set()
        for line in open(LAB / f, encoding="utf-8"):
            line = line.strip()
            if line:
                seen.add(int(line.split()[0]))
        for c in seen:
            by_cls[c].append(f[:-4])

    picked = set()
    for c in range(NC):
        cand = [b for b in by_cls[c] if b not in picked]
        picked.update(random.sample(cand, min(N_PER_CLASS, len(cand))))
    picked = sorted(picked)
    print(f"抽中 {len(picked)} 张")

    if OUT.exists():
        shutil.rmtree(OUT)
    for split, names in (("train", picked[:800]), ("val", picked[800:])):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for b in names:
            for ext in (".jpg", ".png", ".jpeg"):
                src = IMG / (b + ext)
                if src.exists():
                    shutil.copy2(src, OUT / "images" / split / (b + ext))
                    break
            shutil.copy2(LAB / (b + ".txt"), OUT / "labels" / split / (b + ".txt"))
    print(f"train={len(picked[:800])} val={len(picked[800:])}")

    yaml_text = ("path: smoke_data\ntrain: images/train\nval: images/val\ntest: images/val\n\n"
                 "nc: 5\nnames:\n  0: crack\n  1: spalling\n  2: hole\n"
                 "  3: rebar_exposed\n  4: efflorescence\n")
    (BASE / "configs" / "data_smoke.yaml").write_text(yaml_text, encoding="utf-8")

    zp = BASE / "server_pack" / "smoke_data.zip"
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(OUT):
            for f in files:
                p = Path(root) / f
                z.write(p, p.relative_to(BASE))
    print(f"smoke_data.zip: {zp.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
