"""
YOLOv8s 960 分辨率微调训练
================================
在 baseline（640 训练，mAP50 0.4244）基础上，用 960 分辨率微调，
目标是让 hole/corrosion 等小目标（框占图 1~2%）在更高分辨率下看得更清楚。

用法:
    ./.venv/Scripts/python.exe src/train/train_960.py --smoke   # 5 轮冒烟
    ./.venv/Scripts/python.exe src/train/train_960.py --train   # 100 轮微调

预训练: 用 baseline best.pt（640 训练）初始化，微调 100 轮（而非 300 轮从头训）。
"""

import argparse
import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="冒烟测试 5 轮")
    p.add_argument("--train", action="store_true", help="正式微调 100 轮")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    if args.smoke:
        epochs = 5
        name = "yolov26_960_smoke"
        close_mosaic = 0
    else:
        epochs = args.epochs or 100
        name = "yolov26_960_v1"
        close_mosaic = 10

    print(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"epochs={epochs}  batch={args.batch}  imgsz=960  name={name}")

    from ultralytics import YOLO
    # 用 baseline best.pt（640 训练）作为预训练，960 微调
    pretrained = PROJECT_ROOT / "experiments" / "yolov26_baseline_v2" / "weights" / "best.pt"
    print(f"预训练权重: {pretrained} (exists={pretrained.exists()})")
    model = YOLO(str(pretrained))

    model.train(
        data="configs/data_final.yaml",
        epochs=epochs,
        batch=args.batch,
        imgsz=960,
        device=0,
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.0005,       # 微调用更小学习率
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2,
        dropout=0.1,
        patience=30,
        close_mosaic=close_mosaic,
        mosaic=1.0,
        mixup=0.0,
        amp=True,
        seed=0,
        deterministic=True,
        project=str(PROJECT_ROOT / "experiments"),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    print("\n[DONE] 960 微调完成")


if __name__ == "__main__":
    main()
