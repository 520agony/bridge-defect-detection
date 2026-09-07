"""
final_6class seg 模型 v1 正式短训
================================
50 轮训练，拿到第一版真实 mAP，评估 seg 路线天花板与短板类。
"""
import multiprocessing
multiprocessing.freeze_support()

from pathlib import Path
from ultralytics import YOLO

BASE = Path(r'd:\开始做项目\YOLOv26-Project')
WEIGHTS = BASE / 'weights' / 'yolov8s-seg.pt'
DATA = BASE / 'configs' / 'data_final_6class.yaml'
OUT = BASE / 'experiments' / 'v6_seg_v1'


def main():
    print(f"加载预训练: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    print(f"数据: {DATA}")
    print(f"输出: {OUT}")
    print("\n50 轮 v1 短训 ...")

    model.train(
        data=str(DATA),
        epochs=50,
        batch=8,
        imgsz=640,
        device=0,
        workers=2,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        dropout=0.1,
        patience=50,
        close_mosaic=15,
        project=str(OUT.parent),
        name=OUT.name,
        exist_ok=True,
        verbose=True,
        plots=True,
        seed=0,
    )
    print("\n[DONE] 50 轮训练完成")


if __name__ == '__main__':
    main()
