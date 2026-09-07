"""
build_clean_dataset.py - 干净数据集构建工具
==========================================
输入：
  1. 已存在数据集 final_6class_v3（31283 张、6 类、全真 polygon）
  2. 你新下载的数据集（yolov8 segmentation 或 coco segmentation 格式）
  3. bridge crack v1/v2（polygon，源在补充数据 zip）

输出：
  - 在 final_6class_v3 基础上，追加新数据
  - 自动剥 4 点轴对齐矩形（防毒数据混入）
  - 自动合并 6 类（不支持的类警告后跳过）
  - 输出每类 polygon 数 + 总图片数

用法：
  1. 把新数据集 zip 放到 <项目根>/新补充数据/
  2. python scripts/build_clean_dataset.py
  3. 按提示输入新数据集的目录名（如 bridge_crack_v3）

如果新数据是 COCO Segmentation JSON（YOLO 格式行不通的），
也会自动检测并转换。
"""
import shutil
import sys
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
DST = BASE / 'datasets' / 'final_6class_v3'
NEW_DATA_DIR = BASE / '新补充数据'  # 你的新数据放这

# 6 类映射
NAMES = {0: 'crack', 1: 'spalling', 2: 'hole', 3: 'rebar_exposed', 4: 'seepage', 5: 'corrosion'}
# 兼容同义类名（不同数据集的类名差异）
NAME_ALIAS = {
    'crack': 0, 'fracture': 0, 'deep crack': 0,
    'spalling': 1, 'spall': 1,
    'hole': 2, 'cavity': 2, 'pothole': 2, 'pothole asphalt': 2, 'concrete hole': 2,
    'rebar_exposed': 3, 'exposed rebar': 3, 'rebar': 3,
    'seepage': 4, 'infiltration': 4, 'efflorescence': 4, 'concrete leaching': 4,
    'corrosion': 5, 'rust': 5, 'steel corrosion': 5, 'corrode': 5,
}


def is_fake_rect(vals):
    """4 点轴对齐矩形 = bbox→rect 假 mask"""
    if len(vals) != 8:
        return False
    xs, ys = vals[0::2], vals[1::2]
    return (xs[0] == xs[3] and xs[1] == xs[2] and ys[0] == ys[1] and ys[2] == ys[3])


def get_20class_id_map(data_yaml_path):
    """从 20 类数据集的 data.yaml 读 classes，返回 name -> id"""
    import yaml
    cfg = yaml.safe_load(Path(data_yaml_path).read_text(encoding='utf-8'))
    names = cfg.get('names', [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def extract_yolo_seg_zip(zip_path, dst, prefix, target_id_map):
    """解压 YOLO seg 格式 zip，按 split 复制图+写标签（已剥矩形）"""
    import zipfile
    import re
    added_img = added_poly = skipped_unsupported = removed_rect = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # 找 split
        for split_src, split_dst in [('train','train'),('valid','val'),('val','val'),('test','test')]:
            img_dir = f'{split_src}/images'
            lab_dir = f'{split_src}/labels'
            imgs = [n for n in names if n.startswith(img_dir+'/') and n.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
            for iname in imgs:
                # 对应标签
                stem = Path(iname).stem
                lname = f'{lab_dir}/{stem}.txt'
                lines = []
                if lname in names:
                    for line in zf.read(lname).decode('utf-8',errors='replace').strip().splitlines():
                        p = line.split()
                        if not p: continue
                        try: cid = int(float(p[0])); vals = [float(v) for v in p[1:]]
                        except ValueError: continue
                        # 找目标类 ID
                        if cid in target_id_map:
                            target = target_id_map[cid]
                        else:
                            skipped_unsupported += 1
                            continue
                        # bbox→rect 自动检测
                        if len(vals) == 4:
                            cx, cy, w, h = vals
                            if w * h < 0.001: continue
                            vals = [cx-w/2, cy-h/2, cx+w/2, cy-h/2, cx+w/2, cy+h/2, cx-w/2, cy+h/2]
                            removed_rect += 1
                        elif len(vals) >= 6 and len(vals) % 2 == 0:
                            if is_fake_rect(vals):
                                removed_rect += 1
                                continue  # 假矩形，直接丢
                        else:
                            continue
                        lines.append(f"{target} " + ' '.join(f"{v:.6f}" for v in vals))
                        added_poly += 1
                if not lines:
                    continue
                # 写图和标签
                img_data = zf.read(iname)
                new_img_name = prefix + Path(iname).name
                new_lab_name = prefix + stem + '.txt'
                (dst/'images'/split_dst).mkdir(parents=True, exist_ok=True)
                (dst/'labels'/split_dst).mkdir(parents=True, exist_ok=True)
                if (dst/'images'/split_dst/new_img_name).exists():
                    continue  # 防止重名
                (dst/'images'/split_dst/new_img_name).write_bytes(img_data)
                (dst/'labels'/split_dst/new_lab_name).write_text('\n'.join(lines)+'\n', encoding='utf-8')
                added_img += 1
    return added_img, added_poly, skipped_unsupported, removed_rect


def extract_coco_seg_zip(zip_path, dst, prefix, target_id_map):
    """解压 COCO segmentation 格式 zip，把 polygon 绝对坐标归一化后转 YOLO seg"""
    import zipfile, json
    added_img = added_poly = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # 找 annotations json（可能在 train/_annotations.coco.json 等）
        ann_files = [n for n in names if n.endswith('_annotations.coco.json') or n.endswith('annotations.json')]
        if not ann_files:
            return 0, 0, 0, 0
        for ann_file in ann_files:
            data = json.loads(zf.read(ann_file).decode('utf-8'))
            # COCO: images=[{file_name, width, height}], annotations=[{image_id, category_id, segmentation, iscrowd}]
            img_map = {i['id']: i for i in data['images']}
            # cat 映射
            coco_cats = {c['id']: c['name'].lower() for c in data['categories']}
            cat_to_target = {}
            for c_id, c_name in coco_cats.items():
                if c_name in NAME_ALIAS:
                    cat_to_target[c_id] = NAME_ALIAS[c_name]
            # 通常 COCO 一份 json 就是一个 split（train 或 val）
            split = 'val' if 'val' in ann_file else ('test' if 'test' in ann_file else 'train')
            for ann in data['annotations']:
                img_info = img_map.get(ann['image_id'])
                if not img_info: continue
                if ann.get('iscrowd', 0): continue
                segs = ann.get('segmentation', [])
                if not segs: continue
                target = cat_to_target.get(ann['category_id'])
                if target is None: continue
                # 多个 polygon（多段），每个归一化
                W, H = img_info['width'], img_info['height']
                all_lines = []
                for poly in segs:
                    if len(poly) < 6: continue
                    # 归一化
                    pts = []
                    for i in range(0, len(poly), 2):
                        pts.append(f"{poly[i]/W:.6f}")
                        pts.append(f"{poly[i+1]/H:.6f}")
                    all_lines.append(f"{target} " + ' '.join(pts))
                    added_poly += 1
                if not all_lines: continue
                # 找图文件
                img_path = img_info['file_name']
                for p in zf.namelist():
                    if p.endswith(img_path) and p.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
                        img_data = zf.read(p)
                        break
                else:
                    continue
                new_img = prefix + Path(img_path).name
                (dst/'images'/split).mkdir(parents=True, exist_ok=True)
                (dst/'labels'/split).mkdir(parents=True, exist_ok=True)
                if (dst/'images'/split/new_img).exists():
                    continue
                (dst/'images'/split/new_img).write_bytes(img_data)
                (dst/'labels'/(Path(new_img).stem+'.txt')).write_text('\n'.join(all_lines)+'\n', encoding='utf-8')
                added_img += 1
    return added_img, added_poly, 0, 0


def main():
    print('='*60)
    print('build_clean_dataset.py — 把新数据集合并到 final_6class_v3')
    print('='*60)
    if not NEW_DATA_DIR.exists():
        NEW_DATA_DIR.mkdir(exist_ok=True)
        print(f'已创建 {NEW_DATA_DIR}')
        print('请把新数据 zip 放进去，然后重跑本脚本。')
        return
    zip_files = sorted(NEW_DATA_DIR.glob('*.zip'))
    if not zip_files:
        print(f'没有 .zip 文件在 {NEW_DATA_DIR} 下')
        print('请把新数据 zip 放进该目录，重跑。')
        return

    print(f'找到 {len(zip_files)} 个新数据 zip:')
    for z in zip_files:
        print(f'  {z.name}  ({z.stat().st_size//1024//1024}MB)')
    if input('\n确认全部合并到 final_6class_v3？(yes/no): ').strip().lower() not in ('y','yes','是','好'):
        print('已取消')
        return

    # 20 类数据集已内置
    SRC20 = BASE / 'datasets' / '无人机桥梁损伤目标检测YOLO数据集，9978张图像，18类别'
    MAP20_OLD = {4: 0, 5: 0, 16: 1, 9: 3, 3: 5, 18: 5}  # 旧的 6 类映射（已合并过）
    MAP20_NEW = {4: 0, 5: 0, 16: 1, 9: 3, 3: 5, 18: 5, 13: 4}  # 加了 Infiltration(13)->seepage(4)

    # 遍历每个新 zip
    grand_img = grand_poly = grand_unsupp = grand_rect = 0
    for i, zip_path in enumerate(zip_files, 1):
        prefix = f'new{i:02d}_'
        print(f'\n[{i}/{len(zip_files)}] {zip_path.name}  prefix={prefix}')

        # 检测格式：YOLO seg 还是 COCO seg
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            inner = zf.namelist()
        is_coco = any(n.endswith('_annotations.coco.json') for n in inner)
        is_yolo = any(n.startswith('train/labels/') or n.startswith('valid/labels/') or n.startswith('test/labels/') for n in inner)

        if is_yolo:
            # 读 data.yaml 找类别 ID
            yamls = [n for n in inner if n.endswith('data.yaml') or n.endswith('data.yml')]
            target_id_map = {}
            skipped = []
            if yamls:
                import yaml
                cfg = yaml.safe_load(zf.read(yamls[0]).decode('utf-8'))
                names = cfg.get('names', [])
                if isinstance(names, dict):
                    names = [names[k] for k in sorted(names)]
                for i_n, name in enumerate(names):
                    name_l = name.lower().strip()
                    if name_l in NAME_ALIAS:
                        target_id_map[i_n] = NAME_ALIAS[name_l]
                    else:
                        skipped.append(f'{i_n}:{name}')
                print(f'  YOLO seg 模式，类别映射:')
                for src_id, dst_id in target_id_map.items():
                    print(f'    src_id {src_id} ({names[src_id]}) -> dst_id {dst_id} ({NAMES[dst_id]})')
                if skipped:
                    print(f'  跳过的类: {skipped}')
            else:
                print('  [WARN] 没找到 data.yaml，请先确认是 YOLO seg 格式')
                continue
            added_img, added_poly, unsp, rect = extract_yolo_seg_zip(
                zip_path, DST, prefix, target_id_map)
        elif is_coco:
            print(f'  COCO seg 模式')
            added_img, added_poly, unsp, rect = extract_coco_seg_zip(
                zip_path, DST, prefix, target_id_map={})
        else:
            print(f'  [WARN] 未知格式，跳过')
            continue
        grand_img += added_img
        grand_poly += added_poly
        grand_unsupp += unsp
        grand_rect += rect
        print(f'  +{added_img} 图, +{added_poly} polygon, 跳过 {unsp} 框(未支持类), 剥 {rect} 假矩形')

    print(f'\n=== 合并完成 ===')
    print(f'共 +{grand_img} 图, +{grand_poly} polygon, 剥 {grand_rect} 假矩形')
    print()
    c = Counter()
    total = 0
    for split in ['train','val','test']:
        for f in (DST/'labels'/split).glob('*.txt'):
            for line in f.read_text(encoding='utf-8').strip().splitlines():
                p = line.split()
                if p: c[int(float(p[0]))] += 1; total += 1
    print(f'final_6class_v3 现在:')
    for cid in range(6):
        print(f'  {NAMES[cid]:<14} {c[cid]}')
    print(f'  总polygon: {total}')


if __name__ == '__main__':
    main()
