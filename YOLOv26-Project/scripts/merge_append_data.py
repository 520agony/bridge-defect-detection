"""
追加数据并入 final_5class/train（COCO-seg -> YOLO-seg）
=====================================================
来源：D:/开始做项目/追加数据/*.zip（Roboflow coco-segmentation 导出）

合并规则（2026-09 审计结论）：
- Eff(1)：全部，cat1 -> 4
- Eff(2)：只保留与 Eff(3) 无重叠的图（395 张同源只留 Eff(3) 密标版），cat1 -> 4
- Eff(3)：全部，cat3 -> 4；丢弃 cat1('-')/cat2('5') 杂项
- Spalling-detection：全部，cat1 -> 1
- spalling-small：只保留 cat2('spall') -> 1；丢弃 cat1/cat3（棋盘格标定尺！）
- 全部只进 train，不动 val/test
- 多边形归一化 + 裁剪到 [0,1]，不足 3 点则丢弃该实例；整图无实例则写空 txt（背景图）

跑法：.venv/Scripts/python.exe scripts/merge_append_data.py
"""

import json
import os
import shutil
from collections import Counter

UNZIP = r"D:\开始做项目\追加数据\_unzipped"
DST_IMG = r"D:\开始做项目\YOLOv26-Project\datasets\final_5class\images\train"
DST_LAB = r"D:\开始做项目\YOLOv26-Project\datasets\final_5class\labels\train"

# pack -> (目标类id, 保留的源cat id集合, 文件名前缀)
PACKS = {
    "Efflorescence.v1i.coco-segmentation (1)": (4, {1}, "effA_"),
    "Efflorescence.v1i.coco-segmentation (2)": (4, {1}, "effB_"),
    "Efflorescence.v1i.coco-segmentation (3)": (4, {3}, "effC_"),
    "Spalling detection.v1i.coco-segmentation": (1, {1}, "spallA_"),
    "spalling.v1i.coco-segmentation": (1, {2}, "spallB_"),
}


def stem(fn):
    s = os.path.splitext(fn)[0]
    return s.split(".rf.")[0] if ".rf." in s else s


def main():
    # Eff(2)∩Eff(3) 同源：以 Eff(3) 为准
    def pack_stems(pack):
        ss = set()
        for root, _, files in os.walk(os.path.join(UNZIP, pack)):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    ss.add(stem(f))
        return ss

    eff3_stems = pack_stems("Efflorescence.v1i.coco-segmentation (3)")
    skip_stems = eff3_stems  # Eff(2) 中命中这些的一律跳过
    print(f"Eff(3) stems: {len(eff3_stems)}（Eff(2) 命中则跳过）")

    existing = set(os.listdir(DST_IMG)) | set(os.listdir(DST_LAB))
    stat = Counter()
    for pack, (dst_cls, keep_cats, prefix) in PACKS.items():
        pdir = os.path.join(UNZIP, pack)
        for split in ("train", "valid", "test"):
            jp = os.path.join(pdir, split, "_annotations.coco.json")
            if not os.path.exists(jp):
                continue
            with open(jp, encoding="utf-8") as f:
                d = json.load(f)
            imginfo = {im["id"]: im for im in d["images"]}
            anns_by_img = {}
            for a in d["annotations"]:
                anns_by_img.setdefault(a["image_id"], []).append(a)
            for im in d["images"]:
                fn = im["file_name"]
                if pack.startswith("Efflorescence.v1i.coco-segmentation (2)") and stem(fn) in skip_stems:
                    stat["skip_dup_effB"] += 1
                    continue
                Wd, Ht = im["width"], im["height"]
                lines = []
                for a in anns_by_img.get(im["id"], []):
                    if a["category_id"] not in keep_cats:
                        stat["drop_wrong_cat"] += 1
                        continue
                    for seg in a.get("segmentation", []):
                        pts = [(seg[i] / Wd, seg[i + 1] / Ht)
                               for i in range(0, len(seg), 2)]
                        pts = [(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))
                               for x, y in pts]
                        if len(pts) < 3:
                            stat["drop_degenerate"] += 1
                            continue
                        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
                        lines.append(f"{dst_cls} {flat}")
                        stat["kept_instances"] += 1
                # 落盘（加前缀防碰撞）
                src_img = os.path.join(pdir, split, fn)
                base, ext = os.path.splitext(fn)
                if not ext.lower() in (".jpg", ".png", ".jpeg"):
                    ext = ".jpg"
                new_base = prefix + base
                dst_i, dst_l = new_base + ext, new_base + ".txt"
                if dst_i in existing or dst_l in existing:
                    stat["skip_name_collision"] += 1
                    continue
                shutil.copy2(src_img, os.path.join(DST_IMG, dst_i))
                with open(os.path.join(DST_LAB, dst_l), "w") as f:
                    f.write("\n".join(lines))
                existing.add(dst_i)
                existing.add(dst_l)
                stat["kept_images"] += 1
                if not lines:
                    stat["empty_images"] += 1
    print("\n== merge stats ==")
    for k, v in sorted(stat.items()):
        print(f"  {k}: {v}")

    # 清掉 ultralytics 的 labels 缓存，下次训练强制重扫
    for root, _, files in os.walk(os.path.join(DST_LAB, "..")):
        for f in files:
            if f.endswith(".cache"):
                os.remove(os.path.join(root, f))
                print("removed cache:", f)


if __name__ == "__main__":
    main()
