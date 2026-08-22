"""
YOLOv26 强基线训练脚本（干净 7 类数据）
========================================
基于 crack_boost 实证最优配方 + 最终 7 类数据集 (final_7class)

用法:
    # 冒烟测试 (5 轮, 验证 pipeline 无报错、loss 下降)
    ./.venv/Scripts/python.exe src/train/train_final.py --smoke

    # 正式训练 (300 轮, patience=50 自动早停)
    ./.venv/Scripts/python.exe src/train/train_final.py --train

配方要点:
    - yolov8s + AdamW(lr0=0.001) + cos_lr + warmup 3 轮
    - dropout=0.1 轻正则, close_mosaic=15 (最后15轮关mosaic)
    - mixup=0.0 (小目标场景不加重叠混合)
"""

import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/train -> src -> PROJECT_ROOT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="冒烟测试 5 轮")
    p.add_argument("--train", action="store_true", help="正式训练 300 轮")
    p.add_argument("--epochs", type=int, default=None, help="覆盖轮数")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def check_env():
    print("=" * 60)
    print("  环境检查")
    print("=" * 60)
    print(f"PyTorch:   {torch.__version__}  CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU:       {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory // 1024**2
        print(f"VRAM:      {vram} MB")
    else:
        print("[ERROR] GPU 不可用!"); raise SystemExit(1)
    print()


def main():
    args = parse_args()

    if args.smoke:
        epochs = 5
        name = "yolov26_baseline_smoke"
        close_mosaic = 0
    else:
        epochs = args.epochs or 300
        name = "yolov26_baseline_v2"
        close_mosaic = 15

    check_env()

    model_path = PROJECT_ROOT / "yolov8s.pt"
    data_path = PROJECT_ROOT / "configs" / "data_final.yaml"

    print(f"模型:  {model_path.name}  (exists={model_path.exists()})")
    print(f"数据:  {data_path.name}  (exists={data_path.exists()})")
    print(f"epochs={epochs}  batch={args.batch}  imgsz={args.imgsz}  name={name}")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    model.train(
        data=str(data_path),
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
    print("\n[DONE] 训练完成")


if __name__ == "__main__":
    main()
