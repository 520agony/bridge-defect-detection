"""
并入 2381 张真实裂缝 polygon 到 final_6class
=========================================
源：补充数据\新补充数据\bridge crack.v1i / v2i.yolov8.zip（单类 crack，polygon）
目标类：crack（class 0，与源一致）
"""
import zipfile
import shutil
from pathlib import Path

BASE = Path(r'd:\开始做项目\YOLOv26-Project')
DST = BASE / 'datasets' / 'final_6class'
SRC = Path(r'd:\开始做项目\补充数据\新补充数据')
TMP = BASE / 'datasets' / '_crack_tmp'

CRACK_ZIPS = [
    ('bridge crack.v1i.yolov8.zip', 'bc1_'),
    ('bridge crack.v2i.yolov8.zip', 'bc2_'),
]
SPLIT_MAP = [('train', 'train'), ('valid', 'val'), ('val', 'val'), ('test', 'test')]


def bbox_to_polygon(vals):
    cx, cy, w, h = vals
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    return f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"


def main():
    TMP.mkdir(parents=True, exist_ok=True)

    for zipname, prefix in CRACK_ZIPS:
        zp = SRC / zipname
        if not zp.exists():
            print(f"[WARN] 不存在: {zp}")
            continue
        exdir = TMP / zipname.replace('.zip', '')
        exdir.mkdir()
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(exdir)

        added = kept = 0
        for split_src, split_dst in SPLIT_MAP:
            img_dir = exdir / split_src / 'images'
            lab_dir = exdir / split_src / 'labels'
            if not img_dir.exists():
                continue
            dst_img = DST / 'images' / split_dst
            dst_lab = DST / 'labels' / split_dst
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
                        if cid != 0:  # 只要 crack
                            continue
                        if len(vals) == 4:  # bbox → 矩形 polygon
                            lines.append(bbox_to_polygon(vals))
                            kept += 1
                        elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon 原样
                            lines.append('0 ' + ' '.join(f"{v:.6f}" for v in vals))
                            kept += 1
                shutil.copy(img_path, dst_img / new_name)
                (dst_lab / (Path(new_name).stem + '.txt')).write_text(
                    '\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
                added += 1
        print(f"[OK] {zipname}: 添加 {added} 张, {kept} 个 crack 多边形")

    # 汇总
    from collections import Counter
    names = {0: 'crack', 1: 'spalling', 2: 'hole', 3: 'rebar_exposed', 4: 'seepage', 5: 'corrosion'}
    c = Counter()
    for split in ['train', 'val', 'test']:
        for f in (DST / 'labels' / split).glob('*.txt'):
            for line in f.read_text().strip().splitlines():
                p = line.split()
                if p:
                    c[int(float(p[0]))] += 1
    print("\n=== final_6class 并入后每类多边形数 ===")
    for cid in range(6):
        print(f"  {names[cid]:<14} {c[cid]}")


if __name__ == '__main__':
    main()
