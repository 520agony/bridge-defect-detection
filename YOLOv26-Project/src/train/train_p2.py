"""
YOLOv8s-P2 小目标增强训练脚本
================================
在标准 YOLOv8s 基础上加 P2/4 检测头，针对 hole/corrosion 小目标。

用法:
    ./.venv/Scripts/python.exe src/train/train_p2.py --smoke   # 5 轮冒烟测试
    ./.venv/Scripts/python.exe src/train/train_p2.py --train   # 300 轮正式训练

预训练: 用 yolov8s.pt 迁移初始化（backbone + 可匹配的 head 层），新增 P2 层随机初始化。
"""

import argparse
import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="冒烟测试 5 轮")
    p.add_argument("--train", action="store_true", help="正式训练 300 轮")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    if args.smoke:
        epochs = 5
        name = "yolov26_p2_smoke"
        close_mosaic = 0
    else:
        epochs = args.epochs or 300
        name = "yolov26_p2_v1"
        close_mosaic = 15

    print(f"PyTorch {torch.__version__}  CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM {torch.cuda.get_device_properties(0).total_memory//1024**2}MB")
    print(f"epochs={epochs}  batch={args.batch}  imgsz={args.imgsz}  name={name}")

    from ultralytics import YOLO
    # P2 结构 + yolov8s 预训练权重迁移
    model = YOLO("models/yolov8s-p2.yaml").load("yolov8s.pt")
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"模型参数: {n_params/1e6:.2f} M (P2 增强版)")

    model.train(
        data="configs/data_final.yaml",
        epochs=epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        dropout=0.1,
        patience=50,
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
    print("\n[DONE] P2 训练完成")


if __name__ == "__main__":
    main()
