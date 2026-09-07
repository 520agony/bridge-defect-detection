"""
构建 final_6class_v3（真裂缝 polygon 主导）
==========================================
目标：修复 seg 训练中"主干裂缝假 mask（bbox→矩形）"问题

做法：
  1. 复制 final_6class -> final_6class_v3（保留 v1/v2 对照）
  2. 从 TRAIN 剔除"假裂缝"（class 0 + 4 点轴对齐矩形 = 主干 bbox 转的假 mask），
     空图一并删除（避免有裂缝但无标签的负样本）
  3. 并入 20 类数据集（无人机桥梁损伤，全 polygon）的真实多边形：
       Crack(4)+Deep Crack(5) -> crack(0)
       Spalling(16)           -> spalling(1)
       Exposed Rebar(9)       -> rebar_exposed(3)
       Corrosion(3)+Steel Corrosion(18) -> corrosion(5)
     按 Roboflow split 并入（train->train, valid->val, test->test）
"""
import shutil
from pathlib import Path
from collections import Counter

BASE = Path(r'd:\开始做项目\YOLOv26-Project')
SRC6 = BASE / 'datasets' / 'final_6class'
DST = BASE / 'datasets' / 'final_6class_v3'
SRC20 = BASE / 'datasets' / '无人机桥梁损伤目标检测YOLO数据集，9978张图像，18类别'

# 20类 -> 目标类
MAP20 = {4: 0, 5: 0, 16: 1, 9: 3, 3: 5, 18: 5}
SPLIT_MAP = [('train', 'train'), ('valid', 'val'), ('test', 'test')]


def is_fake_rect(vals):
    """4 点轴对齐矩形 = 主干 bbox 转的假 mask；真 polygon 是多点曲线"""
    if len(vals) != 8:
        return False
    xs, ys = vals[0::2], vals[1::2]
    # 矩形角点：x = {x1,x2,x2,x1}, y = {y1,y1,y2,y2}
    return (xs[0] == xs[3] and xs[1] == xs[2] and ys[0] == ys[1] and ys[2] == ys[3])


def main():
    if DST.exists():
        raise SystemExit(f"ERROR: {DST} 已存在，先删除再跑")
    print(f"[1] 复制 {SRC6.name} -> {DST.name} ...")
    shutil.copytree(SRC6, DST)
    print("    完成")

    # ===== 2. 剔 TRAIN 假裂缝 =====
    print("\n[2] 从 TRAIN 剔除假裂缝矩形 + 删除空图 ...")
    removed = empty_del = 0
    train_lab = DST / 'labels' / 'train'
    train_img = DST / 'images' / 'train'
    for f in list(train_lab.glob('*.txt')):
        txt = f.read_text(encoding='utf-8').strip()
        if not txt:
            continue
        new = []
        changed = False
        for line in txt.splitlines():
            p = line.split()
            if not p:
                continue
            cid = int(float(p[0]))
            vals = [float(v) for v in p[1:]]
            if cid == 0 and is_fake_rect(vals):
                changed = True
                removed += 1
                continue
            new.append(line)
        if changed:
            if new:
                f.write_text('\n'.join(new) + '\n', encoding='utf-8')
            else:
                # 图变空（只有假裂缝）-> 删图+标签
                f.unlink()
                img = train_img / (f.stem + '.jpg')
                if img.exists():
                    img.unlink()
                empty_del += 1
    print(f"    剔除假裂缝 {removed} 个, 删除空图 {empty_del} 张")

    # ===== 3. 并入 20 类真 polygon =====
    print(f"\n[3] 并入 20 类数据集真 polygon ...")
    total_added_img = total_added_poly = 0
    for split_src, split_dst in SPLIT_MAP:
        img_dir = SRC20 / split_src / 'images'
        lab_dir = SRC20 / split_src / 'labels'
        if not img_dir.exists():
            continue
        dst_img = DST / 'images' / split_dst
        dst_lab = DST / 'labels' / split_dst
        n_img = n_poly = 0
        for img_path in img_dir.glob('*.jpg'):
            lab_path = lab_dir / (img_path.stem + '.txt')
            if not lab_path.exists():
                continue
            lines = []
            for line in lab_path.read_text(encoding='utf-8').strip().splitlines():
                p = line.split()
                if not p:
                    continue
                try:
                    cid = int(float(p[0]))
                    vals = [float(v) for v in p[1:]]
                except ValueError:
                    continue
                if cid not in MAP20:
                    continue
                if len(vals) < 6 or len(vals) % 2 != 0:
                    continue
                lines.append(f"{MAP20[cid]} " + ' '.join(f"{v:.6f}" for v in vals))
                n_poly += 1
            if not lines:
                continue  # 只含未映射类的图不并入
            # 复制图 + 写标签
            new_name = 'b20_' + img_path.name
            shutil.copy(img_path, dst_img / new_name)
            (dst_lab / (Path(new_name).stem + '.txt')).write_text(
                '\n'.join(lines) + '\n', encoding='utf-8')
            n_img += 1
        total_added_img += n_img
        total_added_poly += n_poly
        print(f"    {split_src} -> {split_dst}: +{n_img} 图, +{n_poly} polygon")
    print(f"    共并入 {total_added_img} 图, {total_added_poly} polygon")

    # ===== 4. 汇总 =====
    print("\n[4] final_6class_v3 总览")
    names = {0: 'crack', 1: 'spalling', 2: 'hole', 3: 'rebar_exposed', 4: 'seepage', 5: 'corrosion'}
    c = Counter()
    for split in ['train', 'val', 'test']:
        for f in (DST / 'labels' / split).glob('*.txt'):
            for line in f.read_text(encoding='utf-8').strip().splitlines():
                p = line.split()
                if p:
                    c[int(float(p[0]))] += 1
    img_n = sum(1 for s in ['train', 'val', 'test'] for _ in (DST / 'images' / s).glob('*.jpg'))
    print(f"  总图片: {img_n}")
    for cid in range(6):
        print(f"    {names[cid]:<14} {c[cid]}")


if __name__ == '__main__':
    main()
