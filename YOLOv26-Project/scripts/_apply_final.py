"""
_apply_final.py — 汇总复核结论 -> confirmed.csv -> 执行隔离
=========================================================
usage:
  python scripts/_apply_final.py build    # 生成 confirmed.csv
  python scripts/_apply_final.py apply    # 执行移动（导入 _review_aggregate.apply）
"""
import csv
import json
import sys
from pathlib import Path

TMP = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\view_review")
AG = TMP / "agents"


def collect_visual(manifest_name, pattern):
    man = json.loads((TMP / manifest_name).read_text(encoding="utf-8"))
    out = {}
    for fp in sorted(AG.glob(pattern)):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ERR] {fp.name}: {e}")
            continue
        n = 0
        for sheet, info in data.items():
            if not isinstance(info, dict) or info.get("clean"):
                continue
            if sheet not in man:
                print(f"[WARN] {fp.name}: sheet {sheet} not in {manifest_name}")
                continue
            for item in info.get("delete", []):
                t = item.get("tile")
                if isinstance(t, int) and 0 <= t < len(man[sheet]):
                    split, stem = man[sheet][t]
                    out[(split, stem)] = (str(item.get("reason", "")), fp.stem)
                    n += 1
        print(f"[{fp.name}] {n} delete-confirmed")
    return out


def build():
    import csv as _csv

    vh = collect_visual("manifest_VH.json", "vh_*.json")
    vo = collect_visual("manifest_VO.json", "vo_*.json")
    print(f"[visual] VH {len(vh)} + VO {len(vo)}")

    # ---- 多信号规则（基于 24 格近原图校准）----
    TMPD = TMP
    manA = json.loads((TMP / "manifest_A.json").read_text(encoding="utf-8"))
    band = {}
    with open(TMP / "seam2_hole.csv", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            band[(r["split"], r["stem"])] = float(r["band"])

    me, rechk = set(), set()
    my = json.loads((TMP / "my_hole_flags.json").read_text(encoding="utf-8"))
    for sh, info in my.items():
        if sh.startswith("_"):
            continue
        for t in info.get("tiles", []):
            me.add(tuple(manA[sh][t]))
    data = json.loads((AG / "hole_recheck.json").read_text(encoding="utf-8"))
    for sh, info in data.items():
        if not isinstance(info, dict) or info.get("clean"):
            continue
        for fl in info.get("flags", []):
            t = fl.get("tile")
            if isinstance(t, int):
                rechk.add(tuple(manA[sh][t]))

    rows = []
    seen = set()

    def put(key, reason, src):
        if key in seen:
            return
        seen.add(key)
        rows.append({"split": key[0], "stem": key[1], "action": "delete",
                     "reason": reason, "src": src})

    for (split, stem), (reason, src) in {**vh, **vo}.items():
        put((split, stem), reason, src)

    # 规则2: 我+复核双标（未视觉确认）
    n2 = 0
    for key in me & rechk:
        if key not in seen:
            put(key, "rule-both", "rule")
            n2 += 1
    print(f"[rule-both] {n2}")

    # 规则3: 单人类标记 且 band>=15
    n3 = 0
    for key in (me | rechk):
        if key not in seen and band.get(key, 0) >= 15.0:
            put(key, "rule-flag+band", "rule")
            n3 += 1
    print(f"[rule-flag+band15] {n3}")

    # 人工已核实 6 格（QC 亲眼确认的 vh_01 漏判拼图）
    qc6 = [("vh_002.jpg", 1), ("vh_002.jpg", 2), ("vh_003.jpg", 0), ("vh_003.jpg", 1),
           ("vh_004.jpg", 0), ("vh_004.jpg", 1)]
    manVH = json.loads((TMP / "manifest_VH.json").read_text(encoding="utf-8"))
    n4 = 0
    for sh, t in qc6:
        key = tuple(manVH[sh][t])
        if key not in seen:
            put(key, "manual-qc", "qc")
            n4 += 1
    print(f"[manual-qc] {n4}")

    # 重复类：算法处理（留一删一）
    nd = 0
    with open(TMP / "candidates.csv", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r["reasons"] != "duplicate":
                continue
            key = (r["split"], r["stem"])
            if key in seen:
                continue
            put(key, "duplicate", r["src"])
            nd += 1
    print(f"[duplicate] {nd}")

    out = TMP / "confirmed.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["split", "stem", "action", "reason", "src"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] confirmed.csv: {len(rows)} rows")
    for s in ("train", "val", "test"):
        print(f"   {s}: {sum(1 for r in rows if r['split'] == s)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    # apply 由外部调用 _review_aggregate.apply
