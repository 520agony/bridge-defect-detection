"""
_review_aggregate.py — 汇总（我的标记 + 审查员标记 + 检测器）→ 候选清单 + 复核接触表
============================================================================
usage:
  python _review_aggregate.py build      # 生成 candidates.csv + 复核表 sheets_V + manifest
  python _review_aggregate.py apply      # (复核完成后) 按 confirmed 清单执行隔离移动
"""
import csv
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TMP = Path(r"C:\Users\梁徐\AppData\Local\Temp\opencode\view_review")
AG = TMP / "agents"
ROOT = Path(r"D:\开始做项目\YOLOv26-Project\datasets\final_5class_clean")
EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
THR_SCORE, THR_GUT, THR_STEP = 95.0, 60.0, 50.0


def img_path(split, stem):
    for e in EXTS:
        p = ROOT / "images" / split / f"{stem}{e}"
        if p.exists():
            return p
    return None


def load_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERR] load {p}: {e}")
        return default if default is not None else {}


def build():
    man = {}
    for f in ("manifest_A.json", "manifest_B.json"):
        man.update(load_json(TMP / f))

    cand = {}

    def add(sheet, tile, src, reason=""):
        if sheet not in man:
            print(f"[WARN] sheet not in manifest: {sheet}")
            return
        arr = man[sheet]
        if not (0 <= tile < len(arr)):
            print(f"[WARN] tile out of range: {sheet}#{tile}")
            return
        split, stem = arr[tile]
        key = f"{split}/{stem}"
        c = cand.setdefault(key, {"src": set(), "reasons": set(), "det": 0.0})
        c["src"].add(src)
        if reason:
            c["reasons"].add(reason)

    # 1) 我的标记
    my = load_json(TMP / "my_hole_flags.json")
    nm = 0
    for sheet, info in my.items():
        if sheet.startswith("_"):
            continue
        for t in info.get("tiles", []):
            add(sheet, t, f"me-{info.get('conf', 'x')}")
            nm += 1
    print(f"[me] {nm}")

    # 2) 审查员标记
    if AG.exists():
        for fp in sorted(AG.glob("*.json")):
            data = load_json(fp)
            na = 0
            for sheet, info in data.items():
                if not isinstance(info, dict) or info.get("clean"):
                    continue
                for fl in info.get("flags", []):
                    t = fl.get("tile")
                    if isinstance(t, int):
                        add(sheet, t, fp.stem.replace(".json", ""), str(fl.get("reason", "")))
                        na += 1
            print(f"[{fp.name}] {na}")

    # 3) 检测器：仅作参考列（不单独进候选，避免噪声淹没复核队列）
    det_lookup = {}
    dp = TMP / "detect_all.csv"
    if dp.exists():
        with open(dp, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    sc = float(row["score"])
                except ValueError:
                    continue
                det_lookup[f"{row['split']}/{row['stem']}"] = sc
    # 检测器独立嫌疑清单（高分但无人标记的，存参考文件）
    suspects = []
    for key, sc in det_lookup.items():
        if sc >= 150.0 and key not in cand:
            split, stem = key.split("/", 1)
            suspects.append({"split": split, "stem": stem, "score": f"{sc:.0f}"})
    suspects.sort(key=lambda r: -float(r["score"]))
    with open(TMP / "detector_suspects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "stem", "score"])
        w.writeheader()
        w.writerows(suspects)
    print(f"[det] 参考列已加载；高分嫌疑(未被人标记) {len(suspects)} -> detector_suspects.csv")

    # 汇总输出（附检测器分数）
    rows = []
    for key, c in cand.items():
        split, stem = key.split("/", 1)
        rows.append({
            "split": split, "stem": stem,
            "n_src": len(c["src"]),
            "src": "|".join(sorted(c["src"])),
            "reasons": "|".join(sorted(c["reasons"])),
            "det_score": f"{det_lookup.get(key, 0):.0f}",
        })
    rows.sort(key=lambda r: (-r["n_src"], r["split"], r["stem"]))
    out = TMP / "candidates.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "stem", "n_src", "src", "reasons", "det_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] candidates: {len(rows)} -> {out}")
    for s in ("train", "val", "test"):
        print(f"   {s}: {sum(1 for r in rows if r['split'] == s)}")

    # 复核接触表（全部候选，按 split 分组）
    outdir = TMP / "sheets_V"
    outdir.mkdir(exist_ok=True)
    for f in outdir.glob("*.jpg"):
        f.unlink()
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    cols, rows_n, tile = 5, 6, 460
    per = cols * rows_n
    vman = {}
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == s]
        n = math.ceil(len(sub) / per)
        for si in range(n):
            chunk = sub[si * per:(si + 1) * per]
            sheet = Image.new("RGB", (tile * cols, tile * rows_n), (20, 20, 24))
            d = ImageDraw.Draw(sheet)
            for i, r in enumerate(chunk):
                p = img_path(r["split"], r["stem"])
                im = None
                if p:
                    try:
                        im = Image.open(p).convert("RGB")
                        im.thumbnail((tile - 8, tile - 8))
                    except Exception:
                        im = None
                x = (i % cols) * tile
                y = (i // cols) * tile
                if im:
                    sheet.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
                d.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(95, 95, 105))
                d.text((x + 8, y + 6), str(i), fill=(255, 235, 0), font=font,
                       stroke_width=2, stroke_fill=(0, 0, 0))
            name = f"v_{s}_{si + 1:03d}.jpg"
            sheet.save(outdir / name, quality=86)
            vman[name] = [[r["split"], r["stem"]] for r in chunk]
    (TMP / "manifest_V.json").write_text(json.dumps(vman, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] sheets_V: {len(vman)} sheets (manifest_V.json)")


def apply(confirmed_path):
    """confirmed.csv: split,stem,action(delete/keep),reason — 执行隔离移动。

    重复(duplicate)特例：同一 split 内先做相似度聚类（MAD<6），每簇只删第 1 张以外的，
    避免把互为重复的两张都删光。
    """
    from collections import defaultdict

    import numpy as np

    qroot = ROOT.parent / "_quarantine_20261008"
    conf = list(csv.DictReader(open(confirmed_path, encoding="utf-8")))
    deletes = [r for r in conf if r.get("action", "").strip() == "delete"]

    # 重复簇处理
    dup_rows = [r for r in deletes if "duplicate" in r.get("reason", "")]
    other_rows = [r for r in deletes if "duplicate" not in r.get("reason", "")]
    dup_only_kept = set()
    clusters = defaultdict(list)
    for r in dup_rows:
        clusters[r["split"]].append(r)

    def small_gray(split, stem):
        p = img_path(split, stem)
        if not p:
            return None
        try:
            im = Image.open(p).convert("L").resize((64, 64))
            return np.asarray(im, dtype=np.float32)
        except Exception:
            return None

    for split, rows in clusters.items():
        vecs = [(r, small_gray(r["split"], r["stem"])) for r in rows]
        vecs = [(r, v) for r, v in vecs if v is not None]
        used = set()
        for i, (ri, vi) in enumerate(vecs):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(vecs)):
                if j in used:
                    continue
                rj, vj = vecs[j]
                mad = float(np.abs(vi - vj).mean())
                if mad < 6.0:
                    group.append(j)
                    used.add(j)
            used.add(i)
            # 簇内按 stem 排序，保留第一张
            grp = sorted(group, key=lambda k: vecs[k][0]["stem"])
            if len(grp) == 1:
                # 找不到近同伙伴：不删（防止误删唯一图）
                dup_only_kept.add((split, vecs[grp[0]][0]["stem"]))
            else:
                for k in grp[1:]:
                    pass  # 保留 vecs[grp[0]]，其余继续删除
                dup_only_kept.add((split, vecs[grp[0]][0]["stem"]))
    # 从 deletes 中剔除每簇保留的第一张
    deletes = [r for r in deletes
               if (r["split"], r["stem"]) not in dup_only_kept]

    moved = 0
    log = []
    for r in deletes:
        split, stem = r["split"], r["stem"]
        ip = img_path(split, stem)
        lp = ROOT / "labels" / split / f"{stem}.txt"
        (qroot / "images" / split).mkdir(parents=True, exist_ok=True)
        (qroot / "labels" / split).mkdir(parents=True, exist_ok=True)
        if ip and ip.exists():
            ip.rename(qroot / "images" / split / ip.name)
        if lp.exists():
            lp.rename(qroot / "labels" / split / lp.name)
        moved += 1
        log.append((split, stem, r.get("reason", ""), r.get("src", "")))
    with open(qroot / "REMOVED.tsv", "a", encoding="utf-8") as f:
        for row in log:
            f.write("\t".join(row) + "\n")
    print(f"[OK] moved {moved} -> {qroot} (duplicate keep-first: {len(dup_only_kept)} kept)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build":
        build()
    elif mode == "apply":
        apply(sys.argv[2])
