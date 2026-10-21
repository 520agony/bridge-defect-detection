r"""
_leak_eval_prep.py — 量化"旧模型开卷优势"
==========================================
依赖: datasets/final_5class/wash_cache.json（wash dry-run 生成的指纹缓存）
逻辑: 家族 = L1基名 ∪ L2md5 ∪ L3dHash精确 的并查集；
      含 {test成员 + train成员} 的家族 → 该 test 图为"旧train有孪生"（affected）
输出:
  TEMP/leak_eval/affected.txt          受影响 test 文件名列表
  TEMP/leak_eval/test_honest/...       干净子集（无孪生）
  TEMP/leak_eval/test_affected/...     受影响子集
  TEMP/leak_eval/{honest,affected}.yaml  评测配置
"""
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"D:\开始做项目\YOLOv26-Project")
OLD = BASE / "datasets" / "final_5class"          # 旧数据（含 wash_cache.json）
CLEAN = BASE / "datasets" / "final_5class_clean"  # 现数据（test 用它的图）
OUT = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\leak_eval")
NAMES = {0: "crack", 1: "spalling", 2: "hole", 3: "rebar_exposed", 4: "efflorescence"}


def base_name(fname):
    n = re.sub(r"\.rf\.[0-9a-f]+", "", fname)
    n = re.sub(r"^n\d+_", "", n)
    return n.lower()


def main():
    cache = json.loads((OLD / "wash_cache.json").read_text(encoding="utf-8"))
    print(f"[cache] {len(cache)} 条")
    keys = [k for k in cache if not k.startswith("_")]
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # L1 基名
    g1 = defaultdict(list)
    for k in keys:
        g1[base_name(k.split("/", 1)[1])].append(k)
    # L2 md5
    g2 = defaultdict(list)
    for k in keys:
        g2[cache[k][1]].append(k)
    # L3 dHash 精确
    g3 = defaultdict(list)
    for k in keys:
        g3[cache[k][2]].append(k)
    for groups in (g1, g2, g3):
        for grp in groups.values():
            for other in grp[1:]:
                union(grp[0], other)

    fam = defaultdict(list)
    for k in keys:
        fam[find(k)].append(k)

    affected = set()
    n_fam_mixed = 0
    for members in fam.values():
        splits = {m.split("/", 1)[0] for m in members}
        if "test" in splits and "train" in splits:
            n_fam_mixed += 1
            for m in members:
                sp, fn = m.split("/", 1)
                if sp == "test":
                    affected.add(fn)
    print(f"[family] 含 test+train 的家族 {n_fam_mixed} 个；受影响 test 图 {len(affected)} 张")

    # 映射到现 clean test
    clean_test = CLEAN / "images" / "test"
    stems = {}
    for p in clean_test.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            stems[p.name] = p
    aff_here = {n for n in affected if n in stems}
    hon_here = [n for n in stems if n not in affected]
    print(f"[clean test] 共 {len(stems)}；受影响 {len(aff_here)}（{len(aff_here)/len(stems)*100:.1f}%）"
          f"；干净子集 {len(hon_here)}")

    OUTPUT = OUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "affected.txt").write_text("\n".join(sorted(aff_here)), encoding="utf-8")

    def build(subset, names, tag):
        d = OUTPUT / subset
        for x in ("images", "labels"):
            (d / x / "test").mkdir(parents=True, exist_ok=True)
        n = 0
        inst = Counter()
        for nm in names:
            src_img = clean_test / nm
            stem = Path(nm).stem
            src_lab = CLEAN / "labels" / "test" / f"{stem}.txt"
            shutil.copy(src_img, d / "images" / "test" / nm)
            if src_lab.exists():
                shutil.copy(src_lab, d / "labels" / "test" / src_lab.name)
                for line in src_lab.read_text(encoding="utf-8").splitlines():
                    p = line.split()
                    if p and int(float(p[0])) in NAMES:
                        inst[int(float(p[0]))] += 1
            n += 1
        (OUTPUT / f"{subset}.yaml").write_text(
            f"path: {str(d).replace(chr(92), '/')}\ntrain: images/test\nval: images/test\n"
            f"test: images/test\n\nnc: 5\nnames:\n" +
            "".join(f"  {i}: {NAMES[i]}\n" for i in range(5)),
            encoding="utf-8")
        print(f"[{tag}] {n} 张 → {d} | 实例: " +
              " ".join(f"{NAMES[c]}={inst[c]}" for c in range(5)))

    build("test_honest", hon_here, "honest")
    build("test_affected", sorted(aff_here), "affected")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
