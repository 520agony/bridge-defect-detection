"""
audit_dataset.py - YOLO seg 数据集全面体检
==========================================
检查项：
  1. 结构：图/标配对、空标签(负样本)、孤儿标签、缺标签
  2. 类别分布：每类实例数/含类图数，train/val/test 占比对比
  3. polygon 质量：短行/非法类/越界坐标/假矩形/退化/粗糙裂缝(<8点)
  4. 目标尺寸：bbox 占图比分布（小目标问题）
  5. 图像分辨率：全量 header 扫描 + 损坏图抽样验证
  6. 跨 split 基名泄漏（L1）

用法（项目根）：
  python scripts/audit_dataset.py --root datasets/final_5class_clean
  python scripts/audit_dataset.py --root datasets/final_5class_clean --fast   # 分辨率抽样3000
输出：
  终端摘要 + <root>/AUDIT_REPORT.md
"""
import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IM_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ["train", "val", "test"]


def percentile(a, p):
    if not a:
        return float("nan")
    a = sorted(a)
    k = (len(a) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)


def base_name(fname):
    n = re.sub(r"\.rf\.[0-9a-f]+", "", fname)
    n = re.sub(r"^n\d+_", "", n)
    return n.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--classes", nargs="+",
                    default=["crack", "spalling", "hole", "rebar_exposed", "efflorescence"])
    ap.add_argument("--fast", action="store_true", help="分辨率只抽样3000张/split")
    ap.add_argument("--dim-sample", type=int, default=0, help="分辨率抽样数(0=全量, --fast=3000)")
    args = ap.parse_args()

    root = Path(args.root)
    assert (root / "images").exists() and (root / "labels").exists(), f"目录结构不对: {root}"
    names = args.classes
    nc = len(names)

    report = []          # markdown lines
    def w(line=""):
        report.append(line)
        print(line)

    w(f"# 数据集体检报告 · {root.name}")
    w()
    w(f"> 生成：audit_dataset.py | 类别（{nc}）：{' / '.join(names)}")
    w()

    # ---------------- 1. 结构 ----------------
    w("## 1. 结构")

    split_images = {}    # split -> set(stem)
    split_labels = {}
    label_files = {}
    w("| split | 图 | 标签 | 空标签 | 缺标签 | 孤儿标签 | 大小 |")
    w("|---|---|---|---|---|---|---|")
    total_size = 0
    for sp in SPLITS:
        idir, ldir = root / "images" / sp, root / "labels" / sp
        imgs = {p.stem: p for p in idir.iterdir() if p.suffix.lower() in IM_EXTS} if idir.exists() else {}
        labs = {p.stem: p for p in ldir.glob("*.txt")} if ldir.exists() else {}
        empty = 0
        for s, lp in labs.items():
            if lp.stat().st_size == 0:
                empty += 1
        missing = len(set(imgs) - set(labs))
        orphan = len(set(labs) - set(imgs))
        sz = sum(p.stat().st_size for p in imgs.values()) + sum(p.stat().st_size for p in labs.values())
        total_size += sz
        split_images[sp] = set(imgs)
        split_labels[sp] = labs
        label_files[sp] = labs
        w(f"| {sp} | {len(imgs)} | {len(labs)} | {empty} | {missing} | {orphan} | {sz/1e9:.2f} GB |")
    w(f"| **合计** | {sum(len(v) for v in split_images.values())} | "
      f"{sum(len(v) for v in split_labels.values())} | | | | **{total_size/1e9:.2f} GB** |")
    w()

    # ---------------- 2+3. 标注解析 ----------------
    per_class_inst = defaultdict(Counter)     # split -> cid -> instances
    per_class_img = defaultdict(Counter)      # split -> cid -> images
    img_has_any = Counter()                   # split -> positive images
    img_inst_count = defaultdict(list)        # split -> instances per positive image
    size_by_class = defaultdict(lambda: defaultdict(list))  # cid -> split -> areas
    flags = defaultdict(Counter)              # split -> flag -> count
    pts_by_class = defaultdict(lambda: defaultdict(list))

    for sp in SPLITS:
        for stem, lp in label_files[sp].items():
            txt = lp.read_text(encoding="utf-8", errors="ignore").strip()
            lines = txt.splitlines() if txt else []
            n_inst = 0
            classes_in_img = set()
            for line in lines:
                p = line.split()
                if len(p) < 7:
                    flags[sp]["short_or_bbox_line"] += 1
                    continue
                try:
                    cid = int(float(p[0]))
                except ValueError:
                    flags[sp]["bad_class_token"] += 1
                    continue
                if not (0 <= cid < nc):
                    flags[sp]["illegal_class"] += 1
                    continue
                vals = [float(v) for v in p[1:]]
                if len(vals) % 2 == 1:
                    flags[sp]["odd_coords"] += 1
                    continue
                xs, ys = vals[0::2], vals[1::2]
                pts_by_class[cid][sp].append(len(xs))
                if len(set(zip(xs, ys))) < 3:
                    flags[sp]["degenerate_lt3"] += 1
                    continue
                oob = any(x < -1e-6 or x > 1 + 1e-6 or y < -1e-6 or y > 1 + 1e-6 for x, y in zip(xs, ys))
                if oob:
                    flags[sp]["oob_coord"] += 1
                if len(xs) == 4:
                    if xs[0] == xs[3] and xs[1] == xs[2] and ys[0] == ys[1] and ys[2] == ys[3]:
                        flags[sp]["fake_rect"] += 1
                if cid == 0 and len(xs) < 8:
                    flags[sp]["crude_crack_lt8"] += 1
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                if area <= 1e-9:
                    flags[sp]["zero_area"] += 1
                    continue
                n_inst += 1
                classes_in_img.add(cid)
                per_class_inst[sp][cid] += 1
                size_by_class[cid][sp].append(area)
            if n_inst > 0:
                img_has_any[sp] += 1
                img_inst_count[sp].append(n_inst)
                for cid in classes_in_img:
                    per_class_img[sp][cid] += 1

    # ---------------- 2. 类别分布 ----------------
    w("## 2. 类别分布（实例数 = 多边形个数；含类图 = 至少含一个该类实例的图）")
    w()
    hdr = "| 类别 | train 实例/图 | val 实例/图 | test 实例/图 | train% | val% | test% |"
    w(hdr)
    w("|---|---|---|---|---|---|---|")
    tot = {sp: sum(per_class_inst[sp].values()) for sp in SPLITS}
    for cid, cname in enumerate(names):
        row = f"| {cname} | {per_class_inst['train'][cid]}/{per_class_img['train'][cid]} " \
              f"| {per_class_inst['val'][cid]}/{per_class_img['val'][cid]} " \
              f"| {per_class_inst['test'][cid]}/{per_class_img['test'][cid]} |"
        for sp in SPLITS:
            row += f" {per_class_inst[sp][cid]/max(tot[sp],1)*100:.1f}% |"
        w(row)
    w(f"| **实例总计** | {tot['train']} | {tot['val']} | {tot['test']} | | | |")
    w()
    w("| split | 正样本图 | 空标签(负样本) | 平均实例/正样本图 | 中位 |")
    w("|---|---|---|---|---|")
    for sp in SPLITS:
        neg = sum(1 for lp in label_files[sp].values() if lp.stat().st_size == 0)
        ic = img_inst_count[sp]
        w(f"| {sp} | {img_has_any[sp]} | {neg} | {sum(ic)/max(len(ic),1):.2f} | {percentile(ic,50):.0f} |")
    w()

    # ---------------- 3. polygon 质量 ----------------
    w("## 3. 标注质量旗标（train/val 应为 0；test 原样保留可能有）")
    w()
    all_flags = sorted({k for sp in SPLITS for k in flags[sp]})
    w("| 旗标 | train | val | test |")
    w("|---|---|---|---|")
    for fl in all_flags:
        w(f"| {fl} | {flags['train'][fl]} | {flags['val'][fl]} | {flags['test'][fl]} |")
    if not all_flags:
        w("| （无） | 0 | 0 | 0 |")
    w()
    w("顶点数分布（min / p5 / 中位 / p95 / max）：")
    w()
    w("| 类别 | split | min | p5 | 中位 | p95 | max |")
    w("|---|---|---|---|---|---|---|")
    for cid, cname in enumerate(names):
        for sp in SPLITS:
            v = pts_by_class[cid][sp]
            if not v:
                continue
            w(f"| {cname} | {sp} | {min(v)} | {percentile(v,5):.0f} | {percentile(v,50):.0f} "
              f"| {percentile(v,95):.0f} | {max(v)} |")
    w()

    # ---------------- 4. 目标尺寸 ----------------
    w("## 4. 目标尺寸（bbox 占图比，小目标关注 <0.1%）")
    w()
    w("| 类别 | train 中位 | <1% | <0.5% | <0.1% | test 中位 | test<0.1% |")
    w("|---|---|---|---|---|---|---|")
    for cid, cname in enumerate(names):
        tr = size_by_class[cid]["train"]
        te = size_by_class[cid]["test"]
        if not tr:
            continue
        w(f"| {cname} | {percentile(tr,50)*100:.2f}% "
          f"| {sum(1 for a in tr if a<0.01)/len(tr)*100:.1f}% "
          f"| {sum(1 for a in tr if a<0.005)/len(tr)*100:.1f}% "
          f"| {sum(1 for a in tr if a<0.001)/len(tr)*100:.1f}% "
          f"| {percentile(te,50)*100:.2f}% "
          f"| {sum(1 for a in te if a<0.001)/max(len(te),1)*100:.1f}% |")
    w()

    # ---------------- 5. 图像分辨率 ----------------
    w("## 5. 图像分辨率（header 扫描）")
    dim_note = "(抽样)" if args.fast else "(全量)"
    w(f"{dim_note}")
    w()
    try:
        from PIL import Image
        w("| split | 扫描数 | W中位 | H中位 | 最小边中位 | <640边占比 | 损坏 |")
        w("|---|---|---|---|---|---|---|")
        rng = random.Random(0)
        for sp in SPLITS:
            files = sorted(split_images[sp])
            sample = rng.sample(files, min(3000, len(files))) if args.fast else files
            corrupt_sample = rng.sample(files, min(300, len(files)))
            paths = {p.stem: p for p in (root / "images" / sp).iterdir() if p.suffix.lower() in IM_EXTS}
            ws, hs, bad = [], [], 0
            for stem in sample:
                try:
                    with Image.open(paths[stem]) as im:
                        x, y = im.size
                    ws.append(x); hs.append(y)
                except Exception:
                    bad += 1
            corrupt2 = 0
            for stem in corrupt_sample:
                try:
                    with Image.open(paths[stem]) as im:
                        im.verify()
                except Exception:
                    corrupt2 += 1
            if ws:
                mins = [min(x, y) for x, y in zip(ws, hs)]
                lt640 = sum(1 for m in mins if m < 640) / len(mins) * 100
                w(f"| {sp} | {len(ws)} | {percentile(ws,50):.0f} | {percentile(hs,50):.0f} "
                  f"| {percentile(mins,50):.0f} | {lt640:.1f}% | {bad}(verify {corrupt2}) |")
            else:
                w(f"| {sp} | 0 | | | | | {bad} |")
        w()
    except ImportError:
        w("（PIL 不可用，跳过分辨率检查）")
        w()

    # ---------------- 6. 基名泄漏 ----------------
    w("## 6. 跨 split 基名泄漏（L1，与 wash 同口径）")
    w()
    bsets = {sp: {base_name(s) for s in split_images[sp]} for sp in SPLITS}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = bsets[a] & bsets[b]
        w(f"- {a} ↔ {b}: **{len(inter)}** 个同名基名")
    if any(bsets[a] & bsets[b] for a, b in [("train", "val"), ("train", "test"), ("val", "test")]):
        ex = sorted(list(bsets["train"] & bsets["val"]))[:5]
        w(f"  - 示例: {ex}")
    w()

    # ---------------- 输出 ----------------
    out = root / "AUDIT_REPORT.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] 报告已写入: {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
