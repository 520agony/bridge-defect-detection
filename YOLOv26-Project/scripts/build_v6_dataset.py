"""
构建 final_6class 数据集（6 类，seg 格式 polygon）
==============================================
源：
  - 主干 final_7class（已改为 6 类 bbox）→ bbox→矩形 polygon（4 角点）
  - 7 个 polygon 补充数据集（_unzipped/）→ 原样用 + 类别映射
  - hole3（bbox）→ bbox→矩形 polygon

类别映射（6 类，honeycomb 已剔除）：
  0 crack 裂缝
  1 spalling 剥落
  2 hole 孔洞
  3 rebar_exposed 露筋
  4 seepage 渗水
  5 corrosion 锈蚀

清理：丢弃 <0.1% 占图的多边形/框
"""
import shutil
from pathlib import Path
from collections import Counter, defaultdict


# 绝对路径（避免 cwd 问题）
BASE = Path(r'd:\开始做项目\YOLOv26-Project')
DST = BASE / 'datasets' / 'final_6class'
BACKBONE_SRC = BASE / 'datasets' / 'final_7class'  # 目录名保留，内部标签已改 6 类
SUPP = Path(r'd:\开始做项目\补充数据\_unzipped')
MIN_AREA = 0.001  # 0.1%

NAMES_6 = {0: 'crack', 1: 'spalling', 2: 'hole', 3: 'rebar_exposed', 4: 'seepage', 5: 'corrosion'}

# 补充数据源：目录、文件名前缀、{源类索引: 目标ID}
SOURCES = [
    ('concrete',                  'conc_',  {0: 0, 1: 1}),                   # crack, spall
    ('concrete_dup',              'cdup_',  {0: 0, 1: 2, 4: 1}),             # crack, hole, spalling (skip honeycombe=2, pitting=3)
    ('concrete_crack_-_spalling', 'ccs_',   {0: 0, 1: 1}),                   # crack, spalling
    ('concrete_spalling',         'csp_',   {1: 1}),                         # spalling (skip "0")
    ('hole1',                     'hole1_', {1: 2}),                         # hole (skip 0, 2)
    ('hole2',                     'hole2_', {1: 2}),                         # Pothole->hole (skip Manhole=0, Bump=2)
    ('hole3',                     'hole3_', {0: 2}),                         # 单类->hole (bbox→polygon)
    ('Spalling',                  'spall_', {0: 0, 1: 1}),                   # crack, spall
]

# 主干类别映射：主干标签已经是 6 类（0-5，之前已去掉 honeycomb+remap），identity 即可
BACKBONE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

SPLIT_MAP = [('train', 'train'), ('valid', 'val'), ('val', 'val'), ('test', 'test')]


def bbox_to_polygon_line(cid, cx, cy, w, h):
    """bbox (class cx cy w h) → polygon (class x1 y1 x2 y2 x3 y3 x4 y4)"""
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    return f"{cid} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"


def poly_area(vals):
    """vals 是一串 x y 坐标，返回多边形外接矩形的面积（用于 tiny 过滤）"""
    xs = vals[0::2]
    ys = vals[1::2]
    if not xs:
        return 0
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def main():
    if DST.exists():
        raise SystemExit(f"ERROR: {DST} 已存在，请先删除再跑")

    # ===== 1. 复制主干 =====
    print(f"[1] 复制主干 {BACKBONE_SRC} -> {DST}")
    shutil.copytree(BACKBONE_SRC, DST)
    print("    完成")

    # ===== 2. 处理主干标签：bbox→矩形 polygon + 类别重映射 =====
    print("\n[2] 处理主干标签（bbox→矩形 polygon，类别重映射）")
    bb_counts = Counter()
    bb_dropped = 0
    for split_src in ['train', 'val', 'test']:
        lab = DST / 'labels' / split_src
        if not lab.exists():
            continue
        for f in lab.glob('*.txt'):
            old = f.read_text().strip().splitlines()
            if not old:
                continue
            new = []
            for line in old:
                p = line.split()
                if not p:
                    continue
                try:
                    cid = int(float(p[0]))
                    vals = [float(v) for v in p[1:]]
                except ValueError:
                    continue
                if cid not in BACKBONE_MAP:
                    bb_dropped += 1
                    continue
                if len(vals) != 4:  # 不是标准 bbox
                    continue
                cx, cy, w, h = vals
                if w * h < MIN_AREA:  # tiny 过滤
                    bb_dropped += 1
                    continue
                new_cid = BACKBONE_MAP[cid]
                new.append(bbox_to_polygon_line(new_cid, cx, cy, w, h))
                bb_counts[new_cid] += 1
            f.write_text('\n'.join(new) + ('\n' if new else ''), encoding='utf-8')

    print("    主干各类多边形数:")
    for cid in range(6):
        print(f"      {NAMES_6[cid]:<14} {bb_counts[cid]}")
    print(f"    丢弃 {bb_dropped} 个（不在映射或 <0.1% 占图）")

    # ===== 3. 处理补充数据 =====
    print("\n[3] 处理补充数据（polygon 原样 / bbox→polygon）")
    for src_name, prefix, idx_map in SOURCES:
        src = SUPP / src_name
        if not src.exists():
            print(f"    [WARN] {src_name} 不存在，跳过")
            continue
        added = kept = dropped = 0
        for split_src, split_dst in SPLIT_MAP:
            img_dir = src / split_src / 'images'
            lab_dir = src / split_src / 'labels'
            dst_img = DST / 'images' / split_dst
            dst_lab = DST / 'labels' / split_dst
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lab.mkdir(parents=True, exist_ok=True)
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob('*'):
                if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                    continue
                new_name = prefix + img_path.name
                lines = []
                lab_path = lab_dir / (img_path.stem + '.txt')
                if lab_path.exists():
                    for line in lab_path.read_text().strip().splitlines():
                        p = line.split()
                        if not p:
                            continue
                        try:
                            cid = int(float(p[0]))
                            vals = [float(v) for v in p[1:]]
                        except ValueError:
                            continue
                        if cid not in idx_map:
                            dropped += 1
                            continue
                        # bbox（4 值）→ 矩形 polygon；polygon（≥6 值偶数）→ 原样
                        if len(vals) == 4:
                            cx, cy, w, h = vals
                            if w * h < MIN_AREA:
                                dropped += 1
                                continue
                            lines.append(bbox_to_polygon_line(idx_map[cid], cx, cy, w, h))
                            kept += 1
                        elif len(vals) >= 6 and len(vals) % 2 == 0:
                            if poly_area(vals) < MIN_AREA:
                                dropped += 1
                                continue
                            lines.append(f"{idx_map[cid]} " + ' '.join(f"{v:.6f}" for v in vals))
                            kept += 1
                        else:
                            dropped += 1
                            continue
                shutil.copy(img_path, dst_img / new_name)
                (dst_lab / (Path(new_name).stem + '.txt')).write_text(
                    '\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8'
                )
                added += 1
        print(f"    {src_name:30s} 添加 {added:>5} 张, 保留 {kept:>6} 个多边形, 丢弃 {dropped}")

    # ===== 4. 汇总 =====
    print("\n[4] final_6class 总览")
    c = Counter()
    img_counts = {}
    for split in ['train', 'val', 'test']:
        n_img = sum(1 for _ in (DST / 'images' / split).glob('*') if _.is_file())
        img_counts[split] = n_img
        for f in (DST / 'labels' / split).glob('*.txt'):
            for line in f.read_text().strip().splitlines():
                p = line.split()
                if p:
                    c[int(float(p[0]))] += 1
    total_img = sum(img_counts.values())
    total_box = sum(c.values())
    print(f"  总图片: {total_img}（train={img_counts['train']}, val={img_counts['val']}, test={img_counts['test']}）")
    print(f"  总多边形: {total_box}")
    for cid in range(6):
        print(f"    {NAMES_6[cid]:<14} {c[cid]}")


if __name__ == '__main__':
    main()
