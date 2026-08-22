"""
构建最终 7 类桥梁病害数据集
================================
最终类目（ID: 名称）:
  0: crack          裂缝
  1: spalling       破损/剥落
  2: honeycomb      蜂窝麻面
  3: hole           孔洞
  4: rebar_exposed  露筋
  5: seepage        渗水/潮湿
  6: corrosion      锈蚀/腐蚀

数据来源:
  1. merged_6class  (官方 6 类, ID 0-5 与上面 0-5 一一对应, 无需重映射)
     更名为 final_7class, 作为主干
  2. 数据集2/Rust腐蚀      -> 类别 6 (corrosion), 原 ID 0 -> 6
  3. 数据集2/Cavity孔洞    -> 类别 3 (hole), 原 ID 0 -> 3
  新增数据按 7:1.5:1.5 划分 train/val/test

本脚本幂等: 若 final_7class 已存在则跳过改名; 重复运行会跳过已存在的文件。
"""

import os
import shutil
import zipfile
import random
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent.parent  # YOLOv26-Project
DATASETS = BASE / "datasets"
SRC_6 = DATASETS / "merged_6class"
DST = DATASETS / "final_7class"
DS2 = Path("d:/开始做项目/数据集2")
RUST_ZIP = DS2 / "Rust腐蚀_2795张_训练包_原图+YOLO标注等2项文件" / "Rust腐蚀_2795张_训练包_原图+YOLO标注.zip"
CAVITY_ZIP = DS2 / "Cavity孔洞_982张_含YOLO标注+可视化.zip"

TRAIN_R, VAL_R = 0.70, 0.15  # 7:1.5:1.5


def log(msg):
    print(msg, flush=True)


def rename_backbone():
    """merged_6class -> final_7class (同盘改名, 瞬间完成, 不复制)"""
    if DST.exists():
        log(f"[SKIP] final_7class 已存在, 跳过改名")
        return
    if not SRC_6.exists():
        log(f"[WARN] 未找到 merged_6class, 从空目录开始")
        DST.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (DST / "images" / split).mkdir(parents=True, exist_ok=True)
            (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
        return
    SRC_6.rename(DST)
    log(f"[OK] merged_6class -> final_7class")


def cleanup_cache():
    for cache in DST.glob("labels/*/*.cache"):
        cache.unlink()
        log(f"[OK] 删除缓存 {cache.name}")
    # 删除可能残留的 empty 标签目录结构不统一问题
    for split in ["train", "val", "test"]:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)


def add_from_zip(zip_path, prefix, class_map):
    """从 zip 提取单类数据, 重映射类别 ID, 按 7:1.5:1.5 分入 final_7class"""
    log(f"\n[ADD] 处理 {zip_path.name} (prefix={prefix}, map={class_map})")
    zf = zipfile.ZipFile(zip_path)
    images = {}   # stem -> zipinfo
    labels = {}   # stem -> zipinfo
    for i in zf.infolist():
        if i.is_dir():
            continue
        parts = i.filename.replace("\\", "/").split("/")
        if len(parts) != 2:
            continue
        stem = os.path.splitext(parts[1])[0]
        if parts[0].lower() == "images" and parts[1].lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            images[stem] = i
        elif parts[0].lower() == "labels" and parts[1].lower().endswith(".txt"):
            labels[stem] = i

    stems = sorted(set(images) & set(labels))
    random.shuffle(stems)
    n = len(stems)
    n_train = int(n * TRAIN_R)
    n_val = int(n * VAL_R)
    counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for idx, stem in enumerate(stems):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"

        img_ext = os.path.splitext(images[stem].filename)[1]
        new_stem = f"{prefix}_{stem}"
        dst_img = DST / "images" / split / f"{new_stem}{img_ext}"
        dst_lbl = DST / "labels" / split / f"{new_stem}.txt"
        if dst_img.exists() and dst_lbl.exists():
            skipped += 1
            continue

        # 读图片字节
        img_data = zf.read(images[stem])
        # 读标签并重映射
        lbl_text = zf.read(labels[stem]).decode("utf-8", errors="replace")
        new_lines = []
        for line in lbl_text.splitlines():
            p = line.split()
            if not p:
                continue
            try:
                old_cls = int(p[0])
            except ValueError:
                continue
            new_cls = class_map.get(old_cls, old_cls)
            new_lines.append(" ".join([str(new_cls)] + p[1:]))

        dst_img.write_bytes(img_data)
        dst_lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        counts[split] += 1

    log(f"  共 {n} 对 (train {counts['train']}, val {counts['val']}, test {counts['test']}, 跳过已存在 {skipped})")
    return counts


def write_data_yaml():
    content = """# 桥梁病害检测 - 最终 7 类数据集配置
# Bridge Defect Detection - Final 7-Class Dataset
# 由 scripts/build_final_dataset.py 生成

path: D:/开始做项目/YOLOv26-Project/datasets/final_7class
train: images/train
val: images/val
test: images/test

nc: 7
names:
  0: crack          # 裂缝
  1: spalling       # 破损/剥落
  2: honeycomb      # 蜂窝麻面
  3: hole           # 孔洞
  4: rebar_exposed  # 露筋
  5: seepage        # 渗水/潮湿
  6: corrosion      # 锈蚀/腐蚀
"""
    out = BASE / "configs" / "data_final.yaml"
    out.write_text(content, encoding="utf-8")
    log(f"\n[OK] 写入配置 {out}")


def verify():
    from collections import Counter
    log("\n" + "=" * 60)
    log("最终数据集统计")
    log("=" * 60)
    names = {0: "crack", 1: "spalling", 2: "honeycomb", 3: "hole",
             4: "rebar_exposed", 5: "seepage", 6: "corrosion"}
    for split in ["train", "val", "test"]:
        img_dir = DST / "images" / split
        lbl_dir = DST / "labels" / split
        n_imgs = len(list(img_dir.glob("*")))
        cnt = Counter()
        bad = 0
        for f in lbl_dir.glob("*.txt"):
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if not p:
                    continue
                try:
                    c = int(p[0])
                except ValueError:
                    bad += 1
                    continue
                if c not in names:
                    bad += 1
                else:
                    cnt[c] += 1
        log(f"\n[{split}] 图片 {n_imgs}, 标注文件 {len(list(lbl_dir.glob('*.txt')))}, 非法标签 {bad}")
        for k in sorted(cnt):
            log(f"   {k} {names[k]:14s}: {cnt[k]}")
        log(f"   合计框数: {sum(cnt.values())}")


if __name__ == "__main__":
    log("=" * 60)
    log("构建最终 7 类桥梁病害数据集")
    log("=" * 60)
    rename_backbone()
    cleanup_cache()
    add_from_zip(RUST_ZIP, "rust", {0: 6})
    add_from_zip(CAVITY_ZIP, "cavity", {0: 3})
    write_data_yaml()
    verify()
    log("\n[DONE] 数据集整理完成")
