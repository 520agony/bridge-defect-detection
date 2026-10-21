"""
wash_independent_test.py - 洗独立测试集（去跨split泄漏）
========================================================
背景：final_5class 的 val/test 与 train 同源泄漏 12.9%/14.7%
（Roboflow同一底图的不同augment进了不同split），val×test还有7.3%重叠。
云上best在这种val上偏乐观，结题前必须洗，否则三轮对比表不可信。

三层去重（优先级 test > val > train，同源组只留一个split）：
  L1 基名同源：去 `.rf.<hash>` / `nXX_` 前缀后同名 → 同一组
  L2 md5精确重复：跨split同md5 → 同一组
  L3 dHash近似重复：汉明距离<=阈值（连拍帧/轻微augment）→ 同一组（BK树加速）

用法（项目根）：
  python scripts/wash_independent_test.py --dry-run        # 只出报告不动数据
  python scripts/wash_independent_test.py --apply           # 写 datasets/final_5class_wash + 新yaml
  python scripts/wash_independent_test.py --apply --no-phash  # 跳过L3（快）
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
NAMES = {0: "crack", 1: "spalling", 2: "hole", 3: "rebar_exposed", 4: "efflorescence"}
PRIO = {"test": 0, "val": 1, "train": 2}  # 数字越小越保留


def base_name(fname):
    n = re.sub(r"\.rf\.[0-9a-f]+", "", fname)
    n = re.sub(r"^n\d+_", "", n)
    return n


def file_md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(p, size=8):
    """PIL实现dHash，返回64位int（无额外依赖）"""
    from PIL import Image
    img = Image.open(p).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(img.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
    return bits


def ham(a, b):
    return bin(a ^ b).count("1")


class BKTree:
    """汉明距离BK树，近似去重加速"""
    def __init__(self, dist):
        self.dist, self.root = dist, None

    def add(self, item):
        if self.root is None:
            self.root = (item, {})
            return
        node = self.root
        while True:
            d = self.dist(item, node[0])
            if d == 0:
                return
            nxt = node[1].get(d)
            if nxt is None:
                node[1][d] = (item, {})
                return
            node = nxt

    def query(self, item, thresh):
        out = []
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            d = self.dist(item, node[0])
            if d <= thresh:
                out.append(node[0])
            for k, child in node[1].items():
                if abs(k - d) <= thresh:
                    stack.append(child)
        return out


def polygon_counts(lab_dir):
    c = Counter()
    for f in lab_dir.glob("*.txt"):
        try:
            for line in f.read_text(encoding="utf-8").strip().splitlines():
                p = line.split()
                if p:
                    c[int(float(p[0]))] += 1
        except Exception:
            pass
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(BASE / "datasets" / "final_5class"))
    ap.add_argument("--dst", default=str(BASE / "datasets" / "final_5class_wash"))
    ap.add_argument("--dry-run", action="store_true", help="只报告不写盘（默认行为，加--apply才写）")
    ap.add_argument("--apply", action="store_true", help="写清洗后数据集+新yaml")
    ap.add_argument("--no-phash", action="store_true", help="跳过L3近似去重")
    ap.add_argument("--phash-thresh", type=int, default=0,
                        help="L3 dHash汉明阈值。默认0（仅跨split精确近似，如同底图不同导出）。"
                             " concrete纹理在阈值>=3时大量误伤，6会删掉29%%训练集，切勿用6")
    ap.add_argument("--hash-only", action="store_true", help="只建指纹缓存就退出（指纹慢时分步跑）")
    args = ap.parse_args()
    write = args.apply

    src = Path(args.src)
    items = {}  # (split, fname) -> dict
    for split in ["train", "val", "test"]:
        for img in (src / "images" / split).iterdir():
            lab = src / "labels" / split / (img.stem + ".txt")
            items[(split, img.name)] = {"path": img, "split": split,
                                        "lab": lab if lab.exists() else None}

    # L1 基名分组
    groups = defaultdict(list)
    for key in items:
        groups[("base", base_name(key[1]))].append(key)

    # 指纹：md5+dHash单遍扫描（8线程，增量缓存；每文件只读一次）
    import concurrent.futures
    import io
    import time
    from PIL import Image
    print("[指纹] md5+dHash 单遍扫描（8线程，增量缓存）...")
    t0 = time.time()
    cache_p = src / "wash_cache.json"
    try:
        hcache = json.loads(cache_p.read_text(encoding="utf-8"))
    except Exception:
        hcache = {}

    def _fp(kv):
        key, it = kv
        rel = f"{key[0]}/{key[1]}"
        try:
            mt = it["path"].stat().st_mtime
        except Exception:
            return key, None
        hit = hcache.get(rel)
        if hit and hit[0] == mt:
            return key, (hit[1], hit[2])
        try:
            data = it["path"].read_bytes()
            mh = hashlib.md5(data).hexdigest()
            img = Image.open(io.BytesIO(data)).convert("L").resize((9, 8), Image.BILINEAR)
            px = list(img.getdata())
            bits = 0
            for r in range(8):
                for c in range(8):
                    bits = (bits << 1) | (1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0)
            hcache[rel] = (mt, mh, bits)
            return key, (mh, bits)
        except Exception:
            return key, None

    md5_of, hashes = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for i, (key, val) in enumerate(ex.map(_fp, items.items())):
            if val is not None:
                md5_of[key], hashes[key] = val
            if (i + 1) % 10000 == 0:
                try:
                    cache_p.write_text(json.dumps(hcache), encoding="utf-8")
                except Exception:
                    pass
                print(f"  ... {i + 1}/{len(items)} {time.time() - t0:.0f}s")
    try:
        cache_p.write_text(json.dumps(hcache), encoding="utf-8")
    except Exception as e:
        print(f"  缓存写失败（忽略）: {e}")
    print(f"  指纹完成 {time.time() - t0:.0f}s：md5 {len(md5_of)}，dhash {len(hashes)}")
    if args.hash_only:
        print("[hash-only] 缓存已落盘，退出。")
        return
    md5_groups = defaultdict(list)
    for key, h in md5_of.items():
        md5_groups[("md5", h)].append(key)

    # 合并L1+L2：union-find
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for key in items:
        parent[key] = key
    for grp in list(groups.values()) + list(md5_groups.values()):
        for other in grp[1:]:
            union(grp[0], other)

    # L3 dHash近似：树建全池，只查val/test（泄漏只与跨split有关）
    near_pairs = 0
    if not args.no_phash:
        t1 = time.time()
        print(f"[L3] BK树（全池建树，只查val/test，阈值<={args.phash_thresh}）...")
        tree = BKTree(ham)
        seen_hash = {}
        for key, h in hashes.items():
            if h not in seen_hash:
                seen_hash[h] = key
                tree.add(h)
        print(f"  建树完成 {time.time() - t1:.0f}s，唯一hash {len(seen_hash)}")
        nq = 0
        for key in items:
            if key[0] == "train" or key not in hashes:
                continue
            for hit in tree.query(hashes[key], args.phash_thresh):
                other = seen_hash[hit]
                if other != key:
                    union(key, other)
                    near_pairs += 1
            nq += 1
            if nq % 2000 == 0:
                print(f"  查询 ... {nq} {time.time() - t1:.0f}s")
        print(f"  查询{nq}张完成 {time.time() - t1:.0f}s，近似命中对数: {near_pairs}")

    # 按组裁决：每组只留优先级最高的split
    fam = defaultdict(list)
    for key in items:
        fam[find(key)].append(key)
    keep_split = {}   # key -> 留在哪个split（=原split则保留，否则移到/删除）
    removals = []     # (fname, from_split, reason, kept_in)
    for members in fam.values():
        splits = sorted({m[0] for m in members}, key=lambda s: PRIO[s])
        if len(splits) == 1:
            continue
        keep = splits[0]
        for m in members:
            if m[0] != keep:
                reasons = []
                if base_name(m[1]) in {base_name(x[1]) for x in members if x[0] == keep}:
                    reasons.append("同源基名")
                if md5_of.get(m) and md5_of[m] in {md5_of.get(x) for x in members if x[0] == keep}:
                    reasons.append("md5相同")
                if not reasons:
                    reasons.append("dHash近似" if not args.no_phash else "同源/md5传递合并")
                removals.append((m[1], m[0], "+".join(reasons), keep))

    rem_by_split = Counter(r[1] for r in removals)
    rem_reason = Counter(r[2] for r in removals)
    print(f"\n[裁决] {len(fam)}组 → 需从train/val移出 {len(removals)} 张: {dict(rem_by_split)}")
    print(f"  原因: {dict(rem_reason)}")

    # 各split剩余 + polygon影响
    before_img, before_poly, after_poly = {}, {}, {}
    for split in ["train", "val", "test"]:
        before_img[split] = sum(1 for k in items if k[0] == split)
        before_poly[split] = polygon_counts(src / "labels" / split)
    removed_keys = {(r[1], r[0]) for r in removals}
    for split in ["train", "val", "test"]:
        c = Counter()
        for (s, name), it in items.items():
            if s != split or (s, name) in removed_keys or it["lab"] is None:
                continue
            try:
                for line in it["lab"].read_text(encoding="utf-8").strip().splitlines():
                    p = line.split()
                    if p:
                        c[int(float(p[0]))] += 1
            except Exception:
                pass
        after_poly[split] = c
    print("\n[影响] 图片: " + str({s: f"{before_img[s]}→{before_img[s]-rem_by_split.get(s,0)}" for s in before_img}))
    for cid in range(5):
        print(f"  {NAMES[cid]:<14} " + " ".join(
            f"{s}:{before_poly[s].get(cid,0)}→{after_poly[s].get(cid,0)}" for s in ["train", "val", "test"]))
    for cid in range(5):
        b, a = before_poly["train"].get(cid, 0), after_poly["train"].get(cid, 0)
        if b and (b - a) / b > 0.4:
            print(f"  [WARN] {NAMES[cid]} train polygon掉{(b-a)/b*100:.0f}%（{b}→{a}）："
                  f"该类独立样本本来就少，之前靠重复撑大。建议补独立新数据，否则该类会欠拟合。")

    report = {"src": str(src), "leak_removed": len(removals),
              "removed_by_split": dict(rem_by_split), "reasons": dict(rem_reason),
              "before_images": before_img,
              "after_images": {s: before_img[s] - rem_by_split.get(s, 0) for s in before_img},
              "before_poly": {s: dict(before_poly[s]) for s in before_poly},
              "after_poly": {s: dict(after_poly[s]) for s in after_poly}}
    if write:
        dst = Path(args.dst)
        if dst.exists():
            raise SystemExit(f"ERROR: {dst} 已存在，先删再跑")
        for split in ["train", "val", "test"]:
            (dst / "images" / split).mkdir(parents=True)
            (dst / "labels" / split).mkdir(parents=True)
        n_copy = 0
        for key, it in items.items():
            if key in removed_keys:
                continue
            shutil.copy(it["path"], dst / "images" / key[0] / key[1])
            if it["lab"] is not None:
                shutil.copy(it["lab"], dst / "labels" / key[0] / (Path(key[1]).stem + ".txt"))
            n_copy += 1
        with open(dst / "removals.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "from_split", "reason", "kept_in"])
            w.writerows(removals)
        with open(dst / "wash_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        yaml_p = BASE / "configs" / "data_final_5class_wash.yaml"
        yaml_p.write_text(
            "# 5类 seg 独立测试集版（wash脚本生成，test/val与train零同源泄漏）\n"
            f"path: {str(dst).replace(chr(92), '/')}\n"
            "train: images/train\nval: images/val\ntest: images/test\n\nnc: 5\nnames:\n"
            "  0: crack\n  1: spalling\n  2: hole\n  3: rebar_exposed\n  4: efflorescence\n",
            encoding="utf-8")
        print(f"\n[写盘] {n_copy} 张 → {dst}\n  removals.csv + wash_report.json + {yaml_p.name}")
    else:
        print("\n[dry-run] 未写盘，加 --apply 才写。")


if __name__ == "__main__":
    main()
