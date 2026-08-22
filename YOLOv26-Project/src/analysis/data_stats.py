"""
数据集标注尺寸分布分析
======================
统计每个类别的 bbox 占图比例分布，用于：
  1. 评估「小目标」问题严重程度（占图越小越难检测）
  2. 验证新增数据是否把目标尺寸拉上来了（新数据中位占比是否 ≥ 目标值）

用法:
    ./.venv/Scripts/python.exe -m src.analysis.data_stats \
        --labels datasets/final_7class/labels \
        --classes crack spalling honeycomb hole rebar_exposed seepage corrosion
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyze(labels_dir, classes):
    stats = defaultdict(list)
    img_count = 0
    for split in ["train", "valid", "val", "test"]:
        d = Path(labels_dir) / split
        if not d.exists():
            continue
        for f in d.glob("*.txt"):
            img_count += 1
            txt = f.read_text().strip()
            if not txt:
                continue
            for line in txt.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                w, h = float(parts[3]), float(parts[4])
                stats[cid].append(w * h)

    print(f"总图片数: {img_count}")
    header = f"{'类别':<16}{'框数':>8}{'中位占比':>10}{'<1%':>7}{'<0.5%':>8}{'<0.1%':>8}{'最大占比':>10}"
    print(header)
    print("-" * len(header))
    for cid in sorted(stats):
        name = classes[cid] if cid < len(classes) else str(cid)
        a = np.array(stats[cid])
        if len(a) == 0:
            continue
        print(
            f"{name:<16}{len(a):>8}{np.median(a) * 100:>9.2f}%"
            f"{(a < 0.01).mean() * 100:>6.1f}%{(a < 0.005).mean() * 100:>7.1f}%"
            f"{(a < 0.001).mean() * 100:>7.1f}%{a.max() * 100:>9.1f}%"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="YOLO 标注目录（含 train/val/test 子目录）")
    ap.add_argument(
        "--classes",
        nargs="+",
        default=["crack", "spalling", "honeycomb", "hole", "rebar_exposed", "seepage", "corrosion"],
        help="类别名列表（按 class_id 顺序）",
    )
    args = ap.parse_args()
    analyze(args.labels, args.classes)


if __name__ == "__main__":
    main()
