r"""
_merge_newdata.py — 6个新zip并入 final_5class_clean（只进train/val，test不动）
==========================================================================
类别映射（目标5类）: crack=0/spalling=1/hole=2/rebar_exposed=3/efflorescence=4
 pack0 Concrete(COCO): exposed_rebar->3, spalling->1, 丢复合类
 pack1 Efflorescence(COCO): efflorescence->4, crack->0
 pack2 Pothole_Seg(COCO): pothole->2
 pack3 SpallRebar(COCO): exposed_rebar->3, spalling->1, 丢复合类
 pack4 pothole-project(COCO): pothole/potholes/high/medium/low->2, 丢object
 pack5 PotholesRoads(YOLO txt): 仅class0的polygon行->2（bbox行/未知class1丢弃）
清洗: 越界clip/退化删/假矩形删/crack<8点删/短边<320跳过/清洗后空图跳过
去重: 审查标记剔除 + md5精确去重(vs现有全量+批次内) + test基名防泄漏
新图命名 nd<k>_<stem>，92/8随机分train/val（seed=8）
用法: python scripts/_merge_newdata.py [--dry-run]
"""
import hashlib
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

DIR = Path(r"D:\开始做项目\当前所需要的数据")
CLEAN = Path(r"D:\开始做项目\YOLOv26-Project\datasets\final_5class_clean")
ND = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\newdata")
DRY = "--dry-run" in sys.argv

ZIPS = [
    "Concrete.v1i.coco-segmentation.zip",
    "Efflorescence.v1i.coco-segmentation.zip",
    "Pothole_Segmentation_YOLOv8.v1i.coco-segmentation.zip",
    "Spalling and exposed rebar.v1i.coco-segmentation.zip",
    "pothole project.v1i.coco-segmentation.zip",
    "Potholes and Roads Instance Segmentation.v1i.yolov8.zip",
]
# 每包COCO类别名映射（小写名 -> 目标id）；pack5走YOLO解析
COCO_MAP = [
    {"exposed_rebar": 3, "spalling": 1},
    {"efflorescence": 4, "crack": 0},
    {"pothole": 2},
    {"exposed_rebar": 3, "spalling": 1},
    {"pothole": 2, "potholes": 2, "high": 2, "medium": 2, "low": 2},
    None,
]
PREFIX = ["nd0", "nd1", "nd2", "nd3", "nd4", "nd5"]
NAMES = {0: "crack", 1: "spalling", 2: "hole", 3: "rebar_exposed", 4: "efflorescence"}


def is_fake_rect(xs, ys):
    if len(xs) != 4:
        return False
    return xs[0] == xs[3] and xs[1] == xs[2] and ys[0] == ys[1] and ys[2] == ys[3]


def base_name(s):
    n = re.sub(r"\.rf\.[0-9a-f]+", "", s)
    n = re.sub(r"^n\d+_", "", n)
    return n.lower()


def clean_poly(cid, poly, W, H, stat):
    """poly: 像素坐标list -> YOLO行或None"""
    if len(poly) < 6 or len(poly) % 2 == 1:
        stat["bad_poly"] += 1
        return None
    xs = [min(1.0, max(0.0, poly[k] / W)) for k in range(0, len(poly), 2)]
    ys = [min(1.0, max(0.0, poly[k] / H)) for k in range(1, len(poly), 2)]
    if len(set(zip(xs, ys))) < 3:
        stat["degenerate"] += 1
        return None
    if (max(xs) - min(xs)) * (max(ys) - min(ys)) <= 1e-9:
        stat["degenerate"] += 1
        return None
    if is_fake_rect(xs, ys):
        stat["fake_rect"] += 1
        return None
    if cid == 0 and len(xs) < 8:
        stat["crude_crack"] += 1
        return None
    return str(cid) + " " + " ".join(f"{x:.6f}" for p_ in zip(xs, ys) for x in p_)


def main():
    # ---- 0) 审查剔除名单 ----
    man = json.loads((ND / "manifest_ND.json").read_text(encoding="utf-8"))
    excluded = set()
    nflags = 0
    for fp in sorted((ND / "agents").glob("ndc_*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        for sheet, info in data.items():
            if not isinstance(info, dict) or info.get("clean"):
                continue
            if sheet not in man:
                print(f"[WARN] {fp.name}: sheet {sheet}不在manifest")
                continue
            for fl in info.get("flags", []):
                t = fl.get("tile")
                if isinstance(t, int) and 0 <= t < len(man[sheet]):
                    excluded.add(tuple(man[sheet][t]))
                    nflags += 1
    print(f"[0] 审查标记 {nflags} -> 去重后剔除 {len(excluded)} 张")

    # ---- 1) 现有库md5 + test基名 ----
    print("[1] 现有数据集md5建库...")
    exist_md5 = set()
    test_bases = set()
    for sp in ("train", "val", "test"):
        d = CLEAN / "images" / sp
        for p in d.iterdir():
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            h = hashlib.md5()
            with open(p, "rb") as f:
                for ch in iter(lambda: f.read(1 << 20), b""):
                    h.update(ch)
            exist_md5.add(h.hexdigest())
            if sp == "test":
                test_bases.add(base_name(p.stem))
    print(f"  md5={len(exist_md5)} test基名={len(test_bases)}")

    rng = random.Random(8)
    stat = Counter()
    pending = []
    pack_stat = Counter()

    for pi, zn in enumerate(ZIPS):
        print(f"[2] 解析 pack{pi} {zn[:40]}...")
        z = zipfile.ZipFile(DIR / zn)
        names = z.namelist()
        if pi <= 4:
            cmap = COCO_MAP[pi]
            for jn in [n for n in names if n.lower().endswith(".json")]:
                try:
                    j = json.loads(z.read(jn))
                except Exception:
                    continue
                if "annotations" not in j or "images" not in j:
                    continue
                idmap = {}
                for c in j.get("categories", []):
                    cn = str(c.get("name", "")).lower()
                    if cn in cmap:
                        idmap[c["id"]] = cmap[cn]
                imginfo = {im["id"]: im for im in j["images"]}
                by_img = {}
                for a in j["annotations"]:
                    if a.get("category_id") in idmap:
                        by_img.setdefault(a["image_id"], []).append(
                            (idmap[a["category_id"]], a.get("segmentation")))
                parent = "/".join(jn.split("/")[:-1])
                for iid, lst in by_img.items():
                    im = imginfo.get(iid)
                    if not im:
                        continue
                    fn = str(im["file_name"]).replace("\\", "/")
                    inner = None
                    for n in names:
                        if n == fn or n.endswith("/" + fn) or n.endswith(fn):
                            inner = n
                            break
                    if inner is None:
                        bn = Path(fn).name.lower()
                        for n in names:
                            if n.startswith(parent) and Path(n).name.lower() == bn:
                                inner = n
                                break
                    if inner is None:
                        stat["img_missing"] += 1
                        continue
                    if (pi, inner) in excluded:
                        stat["excluded_review"] += 1
                        continue
                    raw = z.read(inner)
                    W, H = im.get("width", 0), im.get("height", 0)
                    if not W or not H:
                        # 兜底：读图头
                        try:
                            from PIL import Image as PIM
                            _im = PIM.open(io.BytesIO(raw))
                            W, H = _im.size
                        except Exception:
                            stat["img_broken"] += 1
                            continue
                    if min(W, H) < 320:
                        stat["too_small"] += 1
                        pack_stat[f"p{pi}_too_small"] += 1
                        continue
                    md5 = hashlib.md5(raw).hexdigest()
                    if md5 in exist_md5:
                        stat["dup_md5"] += 1
                        pack_stat[f"p{pi}_dup"] += 1
                        continue
                    lines = []
                    for cid, seg in lst:
                        if not (isinstance(seg, list) and seg and isinstance(seg[0], list)):
                            continue
                        for poly in seg:
                            if not isinstance(poly, list):
                                continue
                            ln = clean_poly(cid, poly, W, H, stat)
                            if ln:
                                lines.append(ln)
                    if not lines:
                        stat["empty_after_clean"] += 1
                        continue
                    if base_name(Path(fn).stem) in test_bases:
                        stat["leak_test"] += 1
                        continue
                    exist_md5.add(md5)
                    pending.append((pi, raw, lines, Path(fn).stem))
                    pack_stat[f"p{pi}_ok"] += 1
                    for ln in lines:
                        stat[f"add_inst_{NAMES[int(ln.split()[0])]}"] += 1
        else:
            # pack5: YOLO txt
            txts = [n for n in names if n.lower().endswith(".txt")
                    and "readme" not in n.lower()]
            for tn in txts:
                # 找对应图
                stem = Path(tn).stem
                inner_img = None
                for e in (".jpg", ".jpeg", ".png"):
                    for n in names:
                        if Path(n).stem == stem and n.lower().endswith(e):
                            inner_img = n
                            break
                    if inner_img:
                        break
                if inner_img is None:
                    stat["img_missing"] += 1
                    continue
                if (pi, inner_img) in excluded:
                    stat["excluded_review"] += 1
                    continue
                raw = z.read(inner_img)
                try:
                    from PIL import Image as PIM
                    _im = PIM.open(io.BytesIO(raw))
                    W, H = _im.size
                except Exception:
                    stat["img_broken"] += 1
                    continue
                if min(W, H) < 320:
                    stat["too_small"] += 1
                    pack_stat["p5_too_small"] += 1
                    continue
                md5 = hashlib.md5(raw).hexdigest()
                if md5 in exist_md5:
                    stat["dup_md5"] += 1
                    pack_stat["p5_dup"] += 1
                    continue
                txt = z.read(tn).decode("utf-8", errors="ignore")
                lines = []
                for line in txt.splitlines():
                    p = line.split()
                    if len(p) < 7 or (len(p) - 1) % 2 == 1:
                        stat["bbox_row"] += 1
                        continue
                    try:
                        cid = int(float(p[0]))
                    except ValueError:
                        continue
                    if cid != 0:  # 未知class1丢弃
                        stat["unknown_cls"] += 1
                        continue
                    vals = [float(v) for v in p[1:]]
                    xs = [min(1.0, max(0.0, x)) for x in vals[0::2]]
                    ys = [min(1.0, max(0.0, y)) for y in vals[1::2]]
                    if len(set(zip(xs, ys))) < 3:
                        stat["degenerate"] += 1
                        continue
                    if (max(xs) - min(xs)) * (max(ys) - min(ys)) <= 1e-9:
                        stat["degenerate"] += 1
                        continue
                    if is_fake_rect(xs, ys):
                        stat["fake_rect"] += 1
                        continue
                    lines.append("2 " + " ".join(f"{x:.6f}" for p_ in zip(xs, ys) for x in p_))
                if not lines:
                    stat["empty_after_clean"] += 1
                    continue
                if base_name(stem) in test_bases:
                    stat["leak_test"] += 1
                    continue
                exist_md5.add(md5)
                pending.append((pi, raw, lines, stem))
                pack_stat["p5_ok"] += 1
                stat["add_inst_hole"] += len(lines)

    print("[3] 待写入", len(pending), "张 | 分包:")
    for pi in range(6):
        print(f"   pack{pi}: ok={pack_stat[f'p{pi}_ok']} dup={pack_stat[f'p{pi}_dup']} small={pack_stat[f'p{pi}_too_small']}")

    print(f"[3] 待写入 {len(pending)} 张")
    for k in sorted(stat):
        print(f"   {k}: {stat[k]}")
    if DRY:
        print("[DRY-RUN]")
        return

    from PIL import Image as PIM
    added = Counter()
    for pi, raw, lines, srcstem in pending:
        split = "val" if rng.random() < 0.08 else "train"
        stem = re.sub(r"\W+", "_", f"{PREFIX[pi]}_{srcstem}")[:80]
        ip = CLEAN / "images" / split / f"{stem}.jpg"
        i = 1
        while ip.exists():
            ip = CLEAN / "images" / split / f"{stem}_{i}.jpg"
            i += 1
        try:
            PIM.open(io.BytesIO(raw)).convert("RGB").save(ip, quality=92)
        except Exception:
            ip.write_bytes(raw)
        (CLEAN / "labels" / split / f"{ip.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        added[split] += 1
    print(f"[4] 写入 train={added['train']} val={added['val']}")
    (ND / "MERGE_REPORT.txt").write_text(
        f"pending={len(pending)} train={added['train']} val={added['val']}\n" +
        "\n".join(f"{k}: {v}" for k, v in sorted(stat.items())), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
