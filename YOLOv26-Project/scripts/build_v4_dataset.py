"""
构建 final_7class_v4（剔除有害数据 + 保留有效数据）
================================================
基于公平对比结论（旧 test 集）：
  - corrosion 近景数据有害（corrosion 0.262→0.242）→ 剔除 3 个 corrosion 数据集
  - pitting→honeycomb 有害（honeycomb 0.252→0.223）→ 剔除 pitting 映射
  - hole 坑洞有效（hole 0.276→0.302）→ 保留 3 个 hole 数据集
  - spalling/crack/honeycomb 补充中性 → 保留

最终 = final_7class 主干 + 6 个补充（无 corrosion/pitting）+ 3 个 hole 数据集

映射采用「源类别索引 → 目标ID」（避免未命名类别的歧义）。
"""

import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # YOLOv26-Project
DATASETS = BASE / "datasets"
SRC_BASE = DATASETS / "final_7class"
DST = DATASETS / "final_7class_v4"
SUPP = Path("d:/开始做项目/补充数据/_unzipped")

MIN_AREA = 0.001  # 0.1%

# 目录名 -> (文件名前缀, {源类别索引: 目标ID})
# 目标ID: 0 crack, 1 spalling, 2 honeycomb, 3 hole, 4 rebar, 5 seepage, 6 corrosion
SOURCES = [
    ("concrete", "conc_", {0: 0, 1: 1}),                    # crack, spall
    ("concrete_dup", "cdup_", {0: 0, 1: 3, 2: 2, 4: 1}),    # crack,hole,honeycombe,spalling（跳过 pitting=3）
    ("Honeycomb_Concrete", "hc_", {0: 2}),                  # Honeycomb
    ("Spalling", "spall_", {0: 0, 1: 1}),                   # crack, spall
    ("concrete_crack_-_spalling", "ccs_", {0: 0, 1: 1}),    # crack, spalling
    ("concrete_spalling", "cspall_", {1: 1}),               # spalling（跳过未命名 "0"）
    ("hole1", "hole1_", {1: 3}),                            # hole
    ("hole2", "hole2_", {1: 3}),                            # Pothole -> hole
    ("hole3", "hole3_", {0: 3}),                            # hole
]

SPLIT_MAP = [("train", "train"), ("valid", "val"), ("test", "test")]


def parse_to_bbox(line):
    """解析一行 YOLO 标签，返回 (源类别索引, cx, cy, w, h) 或 None。
    兼容 bbox（5 字段）和 polygon（>=7 字段，取外接框）。"""
    p = line.split()
    if not p:
        return None
    try:
        cid = int(float(p[0]))
        vals = [float(v) for v in p[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        cx, cy, w, h = vals
        return cid, cx, cy, w, h
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return cid, (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    return None


def main():
    if DST.exists():
        raise SystemExit(f"[ERROR] {DST} 已存在，先删除再重建")
    if not SRC_BASE.exists():
        raise SystemExit(f"[ERROR] 主干 {SRC_BASE} 不存在")

    print(f"复制主干 final_7class -> final_7class_v4 ...")
    shutil.copytree(SRC_BASE, DST)
    print("[OK] 主干复制完成")

    for src_name, prefix, idx_map in SOURCES:
        src = SUPP / src_name
        if not src.exists():
            print(f"[WARN] 源不存在: {src_name}")
            continue
        added = kept = 0
        for split_src, split_dst in SPLIT_MAP:
            img_dir = src / split_src / "images"
            lab_dir = src / split_src / "labels"
            dst_img = DST / "images" / split_dst
            dst_lab = DST / "labels" / split_dst
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lab.mkdir(parents=True, exist_ok=True)
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob("*"):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                new_name = prefix + img_path.name
                lines = []
                lab_path = lab_dir / (img_path.stem + ".txt")
                if lab_path.exists():
                    for line in lab_path.read_text().strip().splitlines():
                        r = parse_to_bbox(line)
                        if r is None:
                            continue
                        cid, cx, cy, w, h = r
                        if cid not in idx_map:
                            continue
                        if w * h < MIN_AREA:
                            continue
                        cx = min(max(cx, 0), 1); cy = min(max(cy, 0), 1)
                        w = min(w, 1); h = min(h, 1)
                        lines.append(f"{idx_map[cid]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                        kept += 1
                shutil.copy(img_path, dst_img / new_name)
                (dst_lab / (Path(new_name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
                added += 1
        print(f"[OK] {src_name}: 添加 {added} 张图片, {kept} 个框")

    # 汇总
    from collections import Counter
    names = {0: "crack", 1: "spalling", 2: "honeycomb", 3: "hole", 4: "rebar_exposed", 5: "seepage", 6: "corrosion"}
    counter = Counter()
    for split in ["train", "val", "test"]:
        for f in (DST / "labels" / split).glob("*.txt"):
            txt = f.read_text().strip()
            if not txt:
                continue
            for line in txt.splitlines():
                p = line.split()
                if p:
                    counter[int(float(p[0]))] += 1
    print("\n=== final_7class_v4 每类框数 ===")
    for cid in range(7):
        print(f"  {names[cid]:<14} {counter[cid]}")


if __name__ == "__main__":
    main()
