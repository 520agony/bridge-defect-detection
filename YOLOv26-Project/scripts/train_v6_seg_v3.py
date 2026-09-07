"""
final_6class_v3 seg 模型正式训练
================================
关键改进（v3）：
  - 主干假裂缝矩形已剔除（crack 只训真 polygon）
  - 并入 20 类无人机桥梁数据集真 polygon（crack+17k / rebar+13k / spalling+6k / corrosion+3.7k）
  - batch 6 防 OOM，workers 0 防宿主内存不足
"""
import multiprocessing
multiprocessing.freeze_support()

from pathlib import Path
from ultralytics import YOLO

BASE = Path(r'd:\开始做项目\YOLOv26-Project')
WEIGHTS = BASE / 'weights' / 'yolov8s-seg.pt'
DATA = BASE / 'configs' / 'data_final_6class_v3.yaml'
OUT = BASE / 'experiments' / 'v6_seg_v3'


def main():
    print(f"加载预训练: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    print(f"数据: {DATA}")
    print(f"输出: {OUT}")
    print("\n50 轮 v3 正式训练 ...")

    model.train(
        data=str(DATA),
        epochs=100,
        batch=6,
        imgsz=640,
        device=0,
        workers=0,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        dropout=0.1,
        patience=50,
        close_mosaic=15,
        overlap_mask=False,
        project=str(OUT.parent),
        name=OUT.name,
        exist_ok=True,
        plots=True,
        seed=0,
    )
    print("\n[DONE] v3 训练完成")


if __name__ == '__main__':
    main()
