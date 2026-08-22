"""
构建 final_7class_v3 数据集（现有 7 类主干 + 补充数据）
====================================================
把现有 final_7class 作为主干，合并 9 个补充数据集：
  - polygon 标签转 bbox（取多边形外接框）
  - 类别名映射到 7 类 ID
  - 清理超小框（面积 < 0.1%）和未映射类别

最终类目（ID）: 0 crack, 1 spalling, 2 honeycomb, 3 hole, 4 rebar_exposed, 5 seepage, 6 corrosion

补充数据源（目录 -> {源类别名: 目标ID}）:
  - concrete:                  crack->0, spall->1
  - concrete_dup:              crack->0, hole->3, honeycombe->2, pitting->2, spalling->1
  - corrosion:                 corrosion->6
  - Corrosion_Dataset:         Corrosion->6
  - Honeycomb_Concrete:        Honeycomb->2
  - Spalling:                  crack->0, spall->1
  - concrete_crack_-_spalling: crack->0, spalling->1
  - concrete_spalling:         spalling->1  (丢弃未命名的 "0" 类)
  - corrosion_detection_dup:   corrosion->6

幂等：final_7class_v3 已存在则跳过主干复制；重复运行会跳过已存在文件。
"""

import shutil
import yaml
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # YOLOv26-Project
DATASETS = BASE / "datasets"
SRC_BASE = DATASETS / "final_7class"
DST = DATASETS / "final_7class_v3"
SUPP = Path("d:/开始做项目/补充数据/_unzipped")

MIN_AREA = 0.001  # 0.1% —— 小于此的框视为不可学习的噪声

# 目录名 -> (文件名前缀, {源类别名: 目标ID})
SOURCES = [
    ("concrete", "conc_", {"crack": 0, "spall": 1}),
    ("concrete_dup", "cdup_", {"crack": 0, "hole": 3, "honeycombe": 2, "pitting": 2, "spalling": 1}),
    ("corrosion", "corr_", {"corrosion": 6}),
    ("Corrosion_Dataset", "corrd_", {"Corrosion": 6}),
    ("Honeycomb_Concrete", "hc_", {"Honeycomb": 2}),
    ("Spalling", "spall_", {"crack": 0, "spall": 1}),
    ("concrete_crack_-_spalling", "ccs_", {"crack": 0, "spalling": 1}),
    ("concrete_spalling", "cspall_", {"spalling": 1}),
    ("corrosion_detection_dup", "cdet_", {"corrosion": 6}),
]

SPLIT_MAP = [("train", "train"), ("valid", "val"), ("test", "test")]


def log(msg):
    print(msg, flush=True)


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


def copy_backbone():
    if DST.exists():
        log(f"[SKIP] {DST.name} 已存在，跳过主干复制")
        return
    if not SRC_BASE.exists():
        log(f"[ERROR] 主干 {SRC_BASE} 不存在")
        raise SystemExit(1)
    log(f"复制主干 final_7class -> final_7class_v3 ...")
    shutil.copytree(SRC_BASE, DST)
    log("[OK] 主干复制完成")


def add_supplement():
    """把补充数据合并进 DST"""
    for src_name, prefix, name_map in SOURCES:
        src = SUPP / src_name
        if not src.exists():
            log(f"[WARN] 源不存在: {src_name}")
            continue

        # 读源 data.yaml 得到 索引->类别名
        yamls = list(src.rglob("data.yaml"))
        if not yamls:
            log(f"[WARN] {src_name} 无 data.yaml，跳过")
            continue
        cfg = yaml.safe_load(yamls[0].read_text(encoding="utf-8"))
        names = cfg.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        # 索引 -> 目标ID
        idx_to_tgt = {}
        for idx, name in enumerate(names):
            if name in name_map:
                idx_to_tgt[idx] = name_map[name]

        added = 0
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
                # 解析 + 映射 + 清理
                lines = []
                lab_path = lab_dir / (img_path.stem + ".txt")
                if lab_path.exists():
                    for line in lab_path.read_text().strip().splitlines():
                        r = parse_to_bbox(line)
                        if r is None:
                            continue
                        cid, cx, cy, w, h = r
                        if cid not in idx_to_tgt:
                            continue
                        if w * h < MIN_AREA:
                            continue
                        cx = min(max(cx, 0), 1); cy = min(max(cy, 0), 1)
                        w = min(w, 1); h = min(h, 1)
                        lines.append(f"{idx_to_tgt[cid]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                # 复制图片 + 写标签（可能为空 = 负样本）
                shutil.copy(img_path, dst_img / new_name)
                (dst_lab / (Path(new_name).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
                added += 1
        log(f"[OK] {src_name}: 添加 {added} 张图片")


def summary():
    """统计最终数据集每类框数"""
    from collections import Counter
    names = {0: "crack", 1: "spalling", 2: "honeycomb", 3: "hole", 4: "rebar_exposed", 5: "seepage", 6: "corrosion"}
    counter = Counter()
    for split in ["train", "val", "test"]:
        lab_dir = DST / "labels" / split
        for f in lab_dir.glob("*.txt"):
            txt = f.read_text().strip()
            if not txt:
                continue
            for line in txt.splitlines():
                p = line.split()
                if p:
                    counter[int(float(p[0]))] += 1
    log("\n=== final_7class_v3 每类框数 ===")
    for cid in range(7):
        log(f"  {names[cid]:<14} {counter[cid]}")


if __name__ == "__main__":
    copy_backbone()
    add_supplement()
    summary()
