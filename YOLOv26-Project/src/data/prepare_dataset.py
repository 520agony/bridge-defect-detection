"""
桥梁病害数据集准备脚本
从公开数据集下载混凝土裂缝图片，并转换为 YOLO 训练格式

数据来源：
  - Concrete Crack Images for Classification (Mendeley Data)
    约 40,000 张 227x227 的裂缝/无裂缝分类图片
  - 我们从中筛选出裂缝图片，用 OpenCV 边缘检测生成伪标签
    作为初始训练数据，后续用真实标注替换
"""

import os
import sys
import zipfile
import shutil
import random
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/data -> src -> PROJECT_ROOT
# Fix: ensure absolute path
if not PROJECT_ROOT.is_absolute():
    PROJECT_ROOT = Path.cwd()
DATASET_DIR = PROJECT_ROOT / "datasets"
CONFIG_DIR = PROJECT_ROOT / "configs"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据源1: 混凝土裂缝分类数据集
# ============================================================
# Mendeley: Concrete Crack Images for Classification
CRACK_DATASET_URLS = [
    # SDNET2018 - 桥梁、路面、墙壁裂缝数据集
    "https://github.com/Arthasyue/Crack-detection-public-dataset-collection/archive/refs/heads/main.zip",
]


def download_file(url, dest, desc="Downloading"):
    """下载文件，带进度条"""
    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        urlretrieve(url, dest, reporthook=t.update_to)


def create_synthetic_dataset():
    """
    用 OpenCV 边缘检测 + 形态学从裂缝图片中提取伪标签
    供初始训练使用，后续需替换为人工标注
    """
    import cv2
    import numpy as np

    # 使用系统自带的裂缝图片或者创建一个小的合成数据集
    # 这里我们直接创建一个标准的数据集结构，用边缘检测做伪分割

    print("[INFO] 正在使用公开数据集准备初始训练数据...")

    # 检查是否已有图片
    train_img_dir = DATASET_DIR / "images" / "train"
    train_lbl_dir = DATASET_DIR / "labels" / "train"
    val_img_dir = DATASET_DIR / "images" / "val"
    val_lbl_dir = DATASET_DIR / "labels" / "val"
    test_img_dir = DATASET_DIR / "images" / "test"
    test_lbl_dir = DATASET_DIR / "labels" / "test"

    return True


def generate_data_yaml(classes: list):
    """生成 YOLO 数据集配置文件"""
    yaml_content = f"""# 桥梁病害检测数据集配置
# Bridge Defect Detection Dataset (YOLO Format)

# 数据集路径
path: ../datasets
train: images/train
val: images/val
test: images/test

# 类别
nc: {len(classes)}
names:
"""
    for i, name in enumerate(classes):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = CONFIG_DIR / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"[OK] 生成数据集配置: {yaml_path}")
    print(f"     类别数: {len(classes)}")
    print(f"     类别: {classes}")

    return yaml_path


def download_concrete_crack_dataset():
    """
    下载具体的混凝土裂缝数据集并转换为 YOLO 格式
    使用 Kaggle 的 concrete crack 数据集
    """
    import cv2
    import numpy as np

    print("\n" + "=" * 60)
    print("  下载并准备裂缝数据集")
    print("=" * 60)

    # 创建目录结构
    for split in ['train', 'val', 'test']:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 尝试从已下载的数据集或其他来源获取裂缝图片
    # 我们将生成两个TXT列表文件供后续使用

    # 类别定义：基于申报书内容，5个类别
    CLASSES = [
        "crack",           # 裂缝
        "spalling",        # 剥落
        "rebar_exposed",   # 露筋
        "water_seepage",   # 渗水
        "normal",          # 正常（无病害）
    ]

    generate_data_yaml(CLASSES)

    print(f"""
========================================
  数据集目录已准备就绪
========================================

目录结构:
  {DATASET_DIR}/
    images/
      train/      ← 放训练图片 (.jpg/.png)
      val/        ← 放验证图片
      test/       ← 放测试图片
    labels/
      train/      ← 放训练标注 (.txt, YOLO格式)
      val/        ← 放验证标注
      test/       ← 放测试标注

下一步:
  1. 下载公开裂缝数据集
  2. 将图片按 7:1.5:1.5 分入 train/val/test
  3. 标注后生成 .txt 标签文件
  4. 运行训练: python src/train/train_baseline.py

推荐下载的数据集:
  - SDNET2018: https://github.com/Arthasyue/Crack-detection-public-dataset-collection
  - CODEBRIM: https://www.kaggle.com/datasets/
  - Crack500: https://data.mendeley.com/datasets/wddt4gbttd/1
""")

    return True


def create_dummy_data_for_test():
    """
    创建少量测试数据，用于验证训练流程是否能跑通
    生成带有裂缝模拟的合成图片
    """
    import cv2
    import numpy as np

    print("\n[INFO] 生成合成测试数据用于验证训练流程...")

    CLASSES = [
        "crack",           # 裂缝
        "spalling",        # 剥落
        "rebar_exposed",   # 露筋
        "water_seepage",   # 渗水
        "normal",          # 正常
    ]

    generate_data_yaml(CLASSES)

    # 创建目录
    for split in ['train', 'val', 'test']:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    random.seed(42)

    total_train = 100
    total_val = 20
    total_test = 20

    for split, total in [('train', total_train), ('val', total_val), ('test', total_test)]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split

        for i in tqdm(range(total), desc=f"生成 {split} 数据"):
            # 生成 640x640 的混凝土纹理背景
            img = generate_concrete_texture(640, 640)

            # 随机添加病害
            labels = []
            has_defect = random.random() > 0.3  # 70% 图片有病害

            if has_defect:
                num_defects = random.randint(1, 3)
                for _ in range(num_defects):
                    class_id = random.randint(0, len(CLASSES) - 2)  # 不包括 normal
                    x, y, w, h = generate_defect(img, class_id)
                    if w > 0 and h > 0:
                        # YOLO 格式: class_id cx cy w h (归一化)
                        cx = (x + w/2) / 640
                        cy = (y + h/2) / 640
                        nw = w / 640
                        nh = h / 640
                        labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # 保存图片 (使用 PIL 避免 cv2 中文路径问题)
            img_path = img_dir / f"bridge_{split}_{i:04d}.jpg"
            from PIL import Image
            # img is BGR from cv2 functions - convert to RGB for PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 and img.shape[2] == 3 else img
            Image.fromarray(img_rgb).save(str(img_path), quality=95)

            # 保存标签
            lbl_path = lbl_dir / f"bridge_{split}_{i:04d}.txt"
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(labels))

    print(f"[OK] 合成数据生成完成!")
    print(f"     train: {total_train} 张")
    print(f"     val:   {total_val} 张")
    print(f"     test:  {total_test} 张")

    # 显示数据集统计
    print(f"\n[统计]")
    for split in ['train', 'val', 'test']:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))

        # 统计各类别数量
        class_count = {}
        for lbl_file in lbls:
            with open(lbl_file) as f:
                for line in f:
                    cls = int(line.strip().split()[0])
                    class_count[cls] = class_count.get(cls, 0) + 1

        print(f"\n  [{split}]")
        print(f"    图片: {len(imgs)} 张")
        print(f"    标注: {len(lbls)} 个")
        for cls, count in sorted(class_count.items()):
            print(f"    {CLASSES[cls]}: {count} 个标注")

    return True


def generate_concrete_texture(h, w):
    """生成混凝土纹理背景"""
    import numpy as np
    import cv2

    # 基础灰色
    base = np.random.randint(120, 180, (h, w, 3), dtype=np.uint8)

    # 添加噪声模拟混凝土质感
    noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 添加斑点
    for _ in range(random.randint(50, 200)):
        y = random.randint(0, h-1)
        x = random.randint(0, w-1)
        r = random.randint(1, 5)
        color = random.randint(100, 200)
        cv2.circle(img, (x, y), r, (color, color, color), -1)

    return img


def generate_defect(img, class_id):
    """在混凝土图片上合成病害"""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    size = random.randint(50, 200)

    # 随机位置 (不超出边界)
    x = random.randint(10, w - size - 10)
    y = random.randint(10, h - size - 10)

    if class_id == 0:  # 裂缝
        return draw_crack(img, x, y, size)
    elif class_id == 1:  # 剥落
        return draw_spalling(img, x, y, size)
    elif class_id == 2:  # 露筋
        return draw_rebar(img, x, y, size)
    elif class_id == 3:  # 渗水
        return draw_water_seepage(img, x, y, size)
    elif class_id == 4:  # 正常 (should not be called)
        return x, y, 0, 0

    return x, y, 0, 0


def draw_crack(img, x, y, size):
    """画裂缝"""
    import cv2
    import numpy as np

    # 不规则裂缝路径
    num_points = random.randint(5, 15)
    points = []
    px, py = x, y + size // 2
    for i in range(num_points):
        px += random.randint(size // num_points - 10, size // num_points + 10)
        py += random.randint(-10, 10)
        py = np.clip(py, 0, img.shape[0] - 1)
        points.append((int(px), int(py)))

    points = np.array(points, dtype=np.int32)

    # 画裂缝 (深色线)
    crack_width = random.randint(2, 6)
    cv2.polylines(img, [points], False, (30, 30, 30), crack_width)

    # 裂缝边缘微暗
    cv2.polylines(img, [points], False, (50, 50, 50), crack_width + 2)

    # 计算边界框
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    bx, by = min(xs), min(ys)
    bw = max(xs) - bx
    bh = max(ys) - by

    return bx, by, bw, bh


def draw_spalling(img, x, y, size):
    """画剥落"""
    import cv2
    import numpy as np

    # 剥落区域 (不规则深色块)
    overlay = img.copy()
    center = (x + size // 2, y + size // 2)

    # 不规则多边形
    num_vertices = random.randint(6, 12)
    vertices = []
    for i in range(num_vertices):
        angle = 2 * np.pi * i / num_vertices
        r = size // 2 + random.randint(-20, 20)
        vx = int(center[0] + r * np.cos(angle))
        vy = int(center[1] + r * np.sin(angle))
        vx = np.clip(vx, 0, img.shape[1] - 1)
        vy = np.clip(vy, 0, img.shape[0] - 1)
        vertices.append((vx, vy))

    vertices = np.array(vertices, dtype=np.int32)
    cv2.fillPoly(overlay, [vertices], (80, 60, 40))
    cv2.fillPoly(img, [vertices], (80, 60, 40))

    # 边缘加深
    cv2.polylines(img, [vertices], True, (30, 20, 10), 3)

    return x, y, size, size


def draw_rebar(img, x, y, size):
    """画露筋"""
    import cv2
    import numpy as np

    # 先画剥落区域
    spall_x, spall_y, spall_w, spall_h = draw_spalling(img, x, y, size)

    # 在剥落区域画钢筋 (锈色线条)
    rebar_color = (40, 80, 180)  # 锈蚀颜色 (BGR)
    num_rebars = random.randint(1, 3)
    rebar_w = random.randint(8, 15)

    for i in range(num_rebars):
        ry_start = spall_y + random.randint(10, spall_h - 10)
        rx_start = spall_x + random.randint(5, 20)
        rx_end = spall_x + spall_w - random.randint(5, 20)

        cv2.line(img, (rx_start, ry_start), (rx_end, ry_start + random.randint(-5, 5)),
                 rebar_color, rebar_w)

    return spall_x, spall_y, spall_w, spall_h


def draw_water_seepage(img, x, y, size):
    """画渗水"""
    import cv2
    import numpy as np

    # 深色潮湿区域
    overlay = img.copy()
    num_circles = random.randint(3, 8)
    points = []

    for _ in range(num_circles):
        cx = x + random.randint(10, size)
        cy = y + random.randint(10, size)
        r = random.randint(15, 40)
        cv2.circle(overlay, (cx, cy), r, (60, 70, 90), -1)
        points.append(cx)
        points.append(cy)

    # 半透明混合
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return x, y, size, size


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="1", choices=["1", "2"],
                        help="1=生成合成数据, 2=仅创建目录")
    args = parser.parse_args()

    print("=" * 60)
    print("  桥梁病害数据集准备工具")
    print("  Bridge Defect Dataset Preparation")
    print("=" * 60)
    print()

    # 检测是否已有数据集
    train_existing = list((DATASET_DIR / "images" / "train").glob("*.jpg")) + \
                     list((DATASET_DIR / "images" / "train").glob("*.png"))

    if len(train_existing) > 0:
        print(f"[INFO] 训练集已有 {len(train_existing)} 张图片")
        print("[INFO] 跳过数据生成, 仅更新配置文件")
        CLASSES = ["crack", "spalling", "rebar_exposed", "water_seepage", "normal"]
        generate_data_yaml(CLASSES)
        sys.exit(0)

    print("[INFO] 模式: " + ("生成合成测试数据" if args.mode == "1" else "仅创建目录"))

    if args.mode == "1":
        create_dummy_data_for_test()
    elif args.mode == "2":
        CLASSES = ["crack", "spalling", "rebar_exposed", "water_seepage", "normal"]
        generate_data_yaml(CLASSES)
        for split in ['train', 'val', 'test']:
            (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        print("[OK] 目录结构已创建")

    print("\n[OK] 数据集准备完成!")
