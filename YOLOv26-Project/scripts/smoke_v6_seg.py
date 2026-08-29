"""
final_6class seg 模型 5 轮冒烟测试
=================================
验证：数据能加载、模型能跑、loss 在降。跑通后启动正式 v1 短训。
"""
import multiprocessing
multiprocessing.freeze_support()

import os
from pathlib import Path
from ultralytics import YOLO

# 绝对路径
BASE = Path(r'd:\开始做项目\YOLOv26-Project')
WEIGHTS = BASE / 'weights' / 'yolov8s-seg.pt'  # ultralytics 首次运行会下载
DATA = BASE / 'configs' / 'data_final_6class.yaml'
OUT = BASE / 'experiments' / 'v6_smoke'


def main():
    print(f"加载预训练: {WEIGHTS}")
    if not WEIGHTS.exists():
        print(f"  权重不存在，将由 ultralytics 自动下载 yolov8s-seg.pt ...")
    model = YOLO(str(WEIGHTS) if WEIGHTS.exists() else 'yolov8s-seg.pt')
    model.to('cpu')  # 避免 device mismatch（与 eval_model.py 同处理）

    print(f"数据: {DATA}")
    print(f"输出: {OUT}")
    print("\n5 轮冒烟 ...")

    model.train(
        data=str(DATA),
        epochs=5,
        batch=8,
        imgsz=640,
        device='cpu',
        workers=0,           # Windows 上避免 dataloader 多进程
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=1,
        amp=True,
        close_mosaic=0,       # 5 轮不关 mosaic
        project=str(OUT.parent),
        name=OUT.name,
        exist_ok=True,
        verbose=True,
        plots=True,
        seed=0,
    )
    print("\n[DONE] 5 轮冒烟完成")


if __name__ == '__main__':
    main()
