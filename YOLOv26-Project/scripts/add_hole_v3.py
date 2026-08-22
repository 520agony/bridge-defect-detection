"""
把 3 个孔洞数据集追加到 final_7class_v3（hole 类）
================================================
hole 类原数据几乎全是小目标（中位占图 0.50%），这 3 个补充数据集目标更大。

源（按类别索引映射，类名多为未命名数字）:
  - hole1: 索引 1 -> hole（3657 框，1.10%）
  - hole2: Pothole(索引1) -> hole（11338 框，1.47%）；Manhole(井盖)/Unmarked Bump(凸起) 丢弃
  - hole3: 索引 0 -> hole（4058 框，2.02%）

⚠️ 域提醒：这批数据疑似「路面坑洞 pothole」，非桥梁混凝土孔洞，训练后须在混凝土 test 集上验证 hole AP。

清理：polygon->bbox、只保留映射到的类别、丢弃 <0.1% 小框。
"""

import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # YOLOv26-Project
DST = BASE / "datasets" / "final_7class_v3"
SUPP = Path("d:/开始做项目/补充数据/_unzipped")

MIN_AREA = 0.001  # 0.1%
HOLE_ID = 3

# 目录名 -> (文件名前缀, {源类别索引: 目标ID})
SOURCES = [
    ("hole1", "hole1_", {1: HOLE_ID}),
    ("hole2", "hole2_", {1: HOLE_ID}),
    ("hole3", "hole3_", {0: HOLE_ID}),
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
    if not DST.exists():
        raise SystemExit(f"[ERROR] {DST} 不存在，请先运行 build_v3_dataset.py")

    for src_name, prefix, idx_map in SOURCES:
        src = SUPP / src_name
        if not src.exists():
            print(f"[WARN] 源不存在: {src_name}")
            continue
        added = 0
        kept_boxes = 0
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
                        kept_boxes += 1
                shutil.copy(img_path, dst_img / new_name)
                (dst_lab / (Path(new_name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
                added += 1
        print(f"[OK] {src_name}: 添加 {added} 张图片, {kept_boxes} 个 hole 框")


if __name__ == "__main__":
    main()
