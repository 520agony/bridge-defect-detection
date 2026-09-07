"""
为 v3 模型重建"真 polygon 评估集"（v3_test_realpoly）
================================================
v3 训练在 v3_train（剔假矩形+并20类）上跑出 0.357 mAP50，
但 eval final_6class_v3 test 时 0.232 mask mAP——暴跌
原因：test GT 仍有 3029 个主干假矩形，模型学的是真细线，IoU 永远到不了 0.5

修：v3 模型在"真 polygon 评估集"上测——拷贝 final_6class_v3，
重建 test 部分（剔假矩形 + 并入 20 类 test 真 polygon）。

train/val 不动（不影响本测试）。
"""
import shutil
from pathlib import Path
from collections import Counter

BASE = Path(r'd:\开始做项目\YOLOv26-Project')
SRC6 = BASE / 'datasets' / 'final_6class_v3'   # v3 训练集
DST = BASE / 'datasets' / 'v3_test_realpoly'
SRC20 = BASE / 'datasets' / '无人机桥梁损伤目标检测YOLO数据集，9978张图像，18类别'

MAP20 = {4: 0, 5: 0, 16: 1, 9: 3, 3: 5, 18: 5}


def is_fake_rect(vals):
    if len(vals) != 8:
        return False
    xs, ys = vals[0::2], vals[1::2]
    return (xs[0] == xs[3] and xs[1] == xs[2] and ys[0] == ys[1] and ys[2] == ys[3])


def main():
    if DST.exists():
        raise SystemExit(f"ERROR: {DST} 已存在")
    print(f"[1] 复制 {SRC6.name} -> {DST.name}")
    shutil.copytree(SRC6, DST)
    print("    完成")

    print("\n[2] 从 TEST 剔除假裂缝矩形 + 删空图")
    removed = empty_del = 0
    test_lab = DST / 'labels' / 'test'
    test_img = DST / 'images' / 'test'
    for f in list(test_lab.glob('*.txt')):
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
                f.unlink()
                img = test_img / (f.stem + '.jpg')
                if img.exists():
                    img.unlink()
                empty_del += 1
    print(f"    剔除假裂缝 {removed} 个, 删除空图 {empty_del} 张")

    print("\n[3] test 集并入 20 类数据集真 polygon")
    test_src = SRC20 / 'test'
    img_dir = test_src / 'images'
    lab_dir = test_src / 'labels'
    dst_img = DST / 'images' / 'test'
    dst_lab = DST / 'labels' / 'test'
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
            if cid not in MAP20 or len(vals) < 6 or len(vals) % 2 != 0:
                continue
            lines.append(f"{MAP20[cid]} " + ' '.join(f"{v:.6f}" for v in vals))
            n_poly += 1
        if not lines:
            continue
        new_name = 'b20_' + img_path.name
        shutil.copy(img_path, dst_img / new_name)
        (dst_lab / (Path(new_name).stem + '.txt')).write_text(
            '\n'.join(lines) + '\n', encoding='utf-8')
        n_img += 1
    print(f"    并入 {n_img} 图, {n_poly} polygon")

    print("\n[4] v3_test_realpoly test 集总览")
    c = Counter()
    for f in (DST / 'labels' / 'test').glob('*.txt'):
        for line in f.read_text(encoding='utf-8').strip().splitlines():
            p = line.split()
            if p:
                c[int(float(p[0]))] += 1
    img_n = sum(1 for _ in (DST / 'images' / 'test').glob('*.jpg'))
    names = {0: 'crack', 1: 'spalling', 2: 'hole', 3: 'rebar_exposed', 4: 'seepage', 5: 'corrosion'}
    print(f"  总图片: {img_n}")
    for cid in range(6):
        print(f"    {names[cid]:<14} {c[cid]}")


if __name__ == '__main__':
    main()
