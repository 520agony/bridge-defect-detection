"""
伪标注脚本 v2 — 用已训练的桥梁病害模型对无标注图片推理生成 YOLO 标签。
"""

import os
from collections import Counter
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "experiments/yolov26_crack_boost/weights/best.pt"
IMAGE_DIR  = "datasets/images/桥梁裂缝500张"

CONFIDENCE = 0.15       # 较低阈值，尽量多检出
IMG_SIZE   = 640
DEVICE     = "cuda"

# 裂缝相关类别 ID（对应 8 类桥梁病害模型）
# 0:corrosion 1:crack 2:degraded_concrete 3:concrete_hole
# 4:moisture 5:pavement_degrade 6:shrinkage_crack 7:sub_shrinkage_crack
CRACK_CLASSES = {1, 6, 7}

# ============================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    print(f"模型类别: {model.names}")
    print(f"图片目录: {IMAGE_DIR}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(exts))
    print(f"图片数量: {len(images)}")

    # 批量推理，ultralytics 自动保存 labels/ 到 runs 下
    results = model.predict(
        source=IMAGE_DIR,
        conf=CONFIDENCE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=False,
        save_txt=True,
        save_conf=False,
        project=IMAGE_DIR,
        name="",
        exist_ok=True,
    )

    # 过滤标注
    auto_label_dir = os.path.join(IMAGE_DIR, "labels")
    total_txt = 0
    kept = 0
    removed_non_crack = 0
    empty = 0
    cls_counter = Counter()

    if not os.path.isdir(auto_label_dir):
        print("错误: labels 目录未生成！")
        return

    for f in sorted(os.listdir(auto_label_dir)):
        if not f.endswith(".txt"):
            continue
        total_txt += 1
        fpath = os.path.join(auto_label_dir, f)
        with open(fpath, "r") as fh:
            lines = fh.readlines()

        if not lines:
            empty += 1
            os.remove(fpath)
            continue

        filtered = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(float(parts[0]))
            if cls_id in CRACK_CLASSES:
                filtered.append(line)
                cls_counter[cls_id] += 1

        if not filtered:
            os.remove(fpath)
            removed_non_crack += 1
        else:
            with open(fpath, "w") as fh:
                fh.writelines(filtered)
            kept += 1

    # 把被模型检测到的图片也复制或用标注覆盖原位置
    print(f"\n{'='*50}")
    print("伪标注完成！")
    print(f"  总图片数:       {len(images)}")
    print(f"  有检测的图片:   {total_txt}")
    print(f"  推理为空的:     {empty}")
    print(f"  非裂缝过滤:     {removed_non_crack}")
    print(f"  最终保留标注:   {kept} 张")
    print(f"  无检测图片数:   {len(images) - kept}")
    print(f"{'='*50}")

    cls_names = model.names
    print("\n出框统计（过滤后）:")
    for cls_id, count in sorted(cls_counter.items()):
        print(f"  {cls_names.get(cls_id, cls_id)}: {count} 个框")

    # 列出完全没有标注的图片
    all_labels = set(
        f.replace(".txt", "")
        for f in os.listdir(auto_label_dir)
        if f.endswith(".txt")
    )
    no_label = [img for img in images if os.path.splitext(img)[0] not in all_labels]
    print(f"\n完全无检测图片数: {len(no_label)} / {len(images)}")


if __name__ == "__main__":
    main()
