"""
build_5class.py - 5 类 polygon 数据集构建（v2_clean 阶段）
========================================================
5 类：0 crack / 1 spalling / 2 hole / 3 rebar_exposed / 4 efflorescence
（去掉 seepage 和 corrosion，加入 efflorescence）

源：
  1. final_6class_v3（31283 张，删 seepage/corrosion 标签）
  2. 18 类无人机桥梁数据集（取 Crack/DeepCrack/Spalling/ExposedRebar/Efflorescence）
  3. 新的数据/ 下的 3 个 COCO segmentation zip

操作：
  - 删 4 (seepage) 和 5 (corrosion) 标签 + 空图
  - 重映射 18 类：4,5->0; 16->1; 9->3; 6->4 (Efflorescence)
  - 合并 3 个新 COCO zip，按 Roboflow split 落到我们的 train/val/test
  - 按图片内容 md5 去重（防 Roboflow 多源重复）
  - 跳过非目标类（seepage / corrosion 不再收）

输入/输出：
  输入：D:/开始做项目/新的数据/*.zip
  输出：D:/开始做项目/YOLOv26-Project/datasets/final_5class
"""
import hashlib
import shutil
import zipfile
import json
from pathlib import Path
from collections import Counter

BASE = Path(r'd:\开始做项目\YOLOv26-Project')
DST = BASE / 'datasets' / 'final_5class'
SRC6 = BASE / 'datasets' / 'final_6class_v3'
SRC20 = BASE / 'datasets' / '无人机桥梁损伤目标检测YOLO数据集，9978张图像，18类别'
NEW_DIR = Path(r'd:\开始做项目\新的数据')

# 5 类
NAMES = {0: 'crack', 1: 'spalling', 2: 'hole', 3: 'rebar_exposed', 4: 'efflorescence'}
ZH = {0: '裂缝', 1: '剥落', 2: '孔洞', 3: '露筋', 4: '泛碱'}

# 18 类中要映射到目标 5 类的源 ID
MAP20 = {4: 0, 5: 0, 16: 1, 9: 3, 6: 4}  # 3,13,18 不再要

# 新数据的 COCO category name -> 目标 ID
NEW_NAME_MAP = {
    'crack': 0, 'Crack': 0,
    'spalling': 1,
    'exposed_rebar': 3, 'spalling-expossed_rebar': 3,  # 复合类只保留 rebar
    'efflorescence': 4,
}


def is_fake_rect(vals):
    """4 点轴对齐矩形 = bbox→rect 假 mask"""
    if len(vals) != 8:
        return False
    xs, ys = vals[0::2], vals[1::2]
    return (xs[0] == xs[3] and xs[1] == xs[2] and ys[0] == ys[1] and ys[2] == ys[3])


def file_md5(path):
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main():
    if DST.exists():
        raise SystemExit(f"ERROR: {DST} 已存在")
    DST.mkdir(parents=True)
    for s in ['train', 'val', 'test']:
        (DST / 'images' / s).mkdir(parents=True)
        (DST / 'labels' / s).mkdir(parents=True)

    # 用 MD5 跟踪已复制图（防 Roboflow 多源重复）
    seen = {}  # md5 -> (split, filename)

    # === 1. 处理 final_6class_v3，删 4,5 标签 ===
    print('[1] 处理 final_6class_v3：删 seepage/corrosion...')
    kept = removed = empty_del = 0
    for split in ['train', 'val', 'test']:
        lab_dir = SRC6 / 'labels' / split
        img_dir = SRC6 / 'images' / split
        for f in lab_dir.glob('*.txt'):
            txt = f.read_text(encoding='utf-8').strip()
            new = []
            changed = False
            if txt:
                for line in txt.splitlines():
                    p = line.split()
                    if not p:
                        continue
                    cid = int(float(p[0]))
                    if cid in (4, 5):  # seepage / corrosion
                        changed = True
                        removed += 1
                        continue
                    new.append(line)
            if changed or txt:
                # 改写标签（可能是空文件，但保留以便后续加数据时一起 merge）
                f_dst = DST / 'labels' / split / f.name
                f_dst.write_text('\n'.join(new) + ('\n' if new else ''), encoding='utf-8')
                # 复制图
                img_src = img_dir / (f.stem + '.jpg')
                if img_src.exists():
                    img_dst = DST / 'images' / split / img_src.name
                    shutil.copy(img_src, img_dst)
                    seen[file_md5(img_dst)] = (split, img_src.name)
                    if not new:
                        empty_del += 1
            elif not new and not txt:
                # 原文件就空：跳过（不要占位）
                pass
    print(f'  删 seepage/corrosion {removed} 个, 无标签空图 {empty_del} 张')

    # === 2. 处理 20 类（已合并过 6 类）===
    print('\n[2] 跳过 20 类（之前已并入 v3 train，重复）')
    print('  提示：如想强制重新合并，删 final_5class/images+labels 重跑')
    # 注意：18 类的 Crack/DeepCrack/Spalling/ExposedRebar/Efflorescence 已经在 v3 train 里了
    # 重复扫会浪费 ~50k 张图的 IO，不重做

    # === 3. 合并 新的数据/ 下的 3 个 COCO zip ===
    print('\n[3] 合并 新的数据/ 下的 COCO zip...')
    new_zips = sorted(NEW_DIR.glob('*.zip'))
    grand_img = grand_poly = grand_dup = 0
    for zip_path in new_zips:
        prefix = f'n{new_zips.index(zip_path):02d}_'
        print(f'\n  {zip_path.name}  prefix={prefix}')

        with zipfile.ZipFile(zip_path) as zf:
            inner = zf.namelist()
            ann_files = sorted([n for n in inner if 'annotation' in n.lower() and n.endswith('.json')])
            for ann_file in ann_files:
                data = json.loads(zf.read(ann_file).decode('utf-8'))
                imgs_meta = {i['id']: i for i in data.get('images', [])}
                ann_list = data.get('annotations', [])
                # cat name -> target_id
                cat_to_target = {}
                for c in data.get('categories', []):
                    name = c['name']
                    if name in NEW_NAME_MAP:
                        cat_to_target[c['id']] = NEW_NAME_MAP[name]

                split = 'train' if 'train' in ann_file else ('valid' if 'valid' in ann_file else ('val' if 'val' in ann_file else 'test'))
                if split == 'valid': split = 'val'

                n_img = n_poly = n_dup = 0
                for ann in ann_list:
                    if ann.get('iscrowd', 0): continue
                    img_info = imgs_meta.get(ann['image_id'])
                    if not img_info: continue
                    target = cat_to_target.get(ann['category_id'])
                    if target is None: continue
                    segs = ann.get('segmentation', [])
                    if not segs: continue
                    W, H = img_info['width'], img_info['height']
                    lines = []
                    skip_poly = False
                    for poly in segs:
                        if len(poly) < 6: continue
                        pts = []
                        for i in range(0, len(poly), 2):
                            pts.append(f"{poly[i] / W:.6f}")
                            pts.append(f"{poly[i + 1] / H:.6f}")
                        vals = [float(x) for x in pts]
                        # 剥假矩形
                        if is_fake_rect(vals):
                            skip_poly = True
                            continue
                        lines.append(f"{target} " + ' '.join(pts))
                        n_poly += 1
                    if skip_poly or not lines:
                        continue

                    # 找图数据
                    fname = img_info['file_name']
                    img_path_in_zip = None
                    for p in inner:
                        if p.endswith(fname) and p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path_in_zip = p
                            break
                    if not img_path_in_zip: continue
                    img_bytes = zf.read(img_path_in_zip)

                    # 去重
                    h = hashlib.md5(img_bytes).hexdigest()
                    if h in seen:
                        n_dup += 1
                        continue
                    seen[h] = (split, fname)

                    new_name = prefix + fname
                    (DST / 'images' / split / new_name).write_bytes(img_bytes)
                    (DST / 'labels' / split / (Path(new_name).stem + '.txt')).write_text(
                        '\n'.join(lines) + '\n', encoding='utf-8')
                    n_img += 1
                print(f'    [{split}] +{n_img} 图, +{n_poly} polygon, 跳过重复 {n_dup}')
                grand_img += n_img
                grand_poly += n_poly
                grand_dup += n_dup
    print(f'\n  共合并 +{grand_img} 图, +{grand_poly} polygon, 去重 {grand_dup}')

    # === 4. 汇总 ===
    print('\n[4] final_5class 汇总')
    c = Counter()
    total_img = 0
    for split in ['train', 'val', 'test']:
        n = 0
        for f in (DST / 'labels' / split).glob('*.txt'):
            for line in f.read_text(encoding='utf-8').strip().splitlines():
                p = line.split()
                if p: c[int(float(p[0]))] += 1
            n += 1
        total_img += sum(1 for _ in (DST / 'images' / split).glob('*.jpg'))
    print(f'  总图片: {total_img}  总 polygon: {sum(c.values())}')
    for cid in range(5):
        print(f'    {NAMES[cid]:<14} ({ZH[cid]:<4})  {c[cid]}')


if __name__ == '__main__':
    main()
