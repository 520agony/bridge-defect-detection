"""
YOLOv26 改进版训练脚本（消融实验）
==================================
用于验证三个改进模块的效果：
  - C2fLite（轻量化 backbone）
  - HDDP（空洞金字塔池化）
  - CoordinateAttention（位置注意力）

与基线（yolov8s，未改进）做对比，形成论文「模型改进」章节。

使用方式：
    python src/train/train_improved.py            # 训练改进版
    python src/train/train_improved.py --check    # 仅检查环境
    python src/train/train_improved.py --val      # 仅验证
"""

import os
import sys
from pathlib import Path

# ============================================================
# 路径设置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # 确保 src 可导入
os.chdir(str(PROJECT_ROOT))

import torch

# ============================================================
# 关键：先注册自定义模块，再导入 ultralytics
# ============================================================
import src.models.register_improvements  # noqa: F401  注册 C2fLite/HDDP/CA

from ultralytics import YOLO  # noqa: E402
import yaml  # noqa: E402


# ============================================================
# 配置
# ============================================================
IMPROVED_YAML = "models/yolov26_improved.yaml"   # 改进版架构
BASELINE_YAML = "yolov8s.yaml"                    # 基线（对比用）
DATA_YAML = "configs/data.yaml"
MODEL_NAME = "yolov26_improved_v1"

CONFIG = dict(
    epochs=300,
    batch=4,            # 内存紧张，降到4
    imgsz=640,
    device=0,
    workers=0,          # Windows 必须为 0
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    # 数据增强（沿用已验证的保守配置）
    hsv_h=0.01, hsv_s=0.5, hsv_v=0.3,
    degrees=5.0, translate=0.1, scale=0.3,
    shear=0.0, fliplr=0.5, perspective=0.0,
    mosaic=0.3, mixup=0.0, copy_paste=0.0, erasing=0.0,
    dropout=0.0,
    patience=80,
    close_mosaic=10,
    multi_scale=False,
    amp=False,          # RTX 5060 cuDNN 稳定性
    save_period=10,     # 内存优化：减少保存频率
    plots=False,        # 关闭预览图，省内存
    iou=0.7, box=7.5, dfl=1.5, cls_pw=1.0,
)


def check_env():
    print("=" * 60)
    print("  YOLOv26 改进版 — 环境检查")
    print("=" * 60)
    print(f"Python:   {sys.version.split()[0]}")
    print(f"PyTorch:  {torch.__version__}  CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
        print(f"GPU:      {gpu} ({vram} MiB)")
    else:
        print("GPU:      无（CPU 模式，仅用于架构验证）")
    print(f"模型 YAML: {IMPROVED_YAML}")

    # 验证改进 YAML 能被正确解析（CPU 构建，不占 GPU）
    print("\n构建改进模型验证架构...")
    model = YOLO(IMPROVED_YAML)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  改进版参数量: {n_params / 1e6:.2f} M")
    print("[OK] 改进模块已正确接入，可以训练")
    print("=" * 60)
    return model


def train():
    print("\n" + "=" * 60)
    print("  YOLOv26 改进版训练")
    print("  C2f-Lite + HDDP + Coordinate Attention")
    print("=" * 60)

    # 先构建改进版架构，再用 .load() 迁移 yolov8s.pt 预训练权重（匹配层自动迁移）
    pretrained = PROJECT_ROOT / "yolov8s.pt"
    model = YOLO(IMPROVED_YAML)
    if pretrained.exists():
        print(f"从预训练权重迁移: {pretrained.name}")
        model = model.load(str(pretrained))
    else:
        print("无预训练权重，从零初始化改进架构")

    results = model.train(
        data=DATA_YAML,
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch"],
        imgsz=CONFIG["imgsz"],
        device=CONFIG["device"],
        workers=CONFIG["workers"],
        optimizer=CONFIG["optimizer"],
        lr0=CONFIG["lr0"], lrf=CONFIG["lrf"],
        momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"],
        warmup_epochs=CONFIG["warmup_epochs"],
        warmup_momentum=CONFIG["warmup_momentum"],
        warmup_bias_lr=CONFIG["warmup_bias_lr"],
        cos_lr=CONFIG["cos_lr"],
        hsv_h=CONFIG["hsv_h"], hsv_s=CONFIG["hsv_s"], hsv_v=CONFIG["hsv_v"],
        degrees=CONFIG["degrees"], translate=CONFIG["translate"], scale=CONFIG["scale"],
        shear=CONFIG["shear"], fliplr=CONFIG["fliplr"], perspective=CONFIG["perspective"],
        mosaic=CONFIG["mosaic"], mixup=CONFIG["mixup"],
        copy_paste=CONFIG["copy_paste"], erasing=CONFIG["erasing"],
        dropout=CONFIG["dropout"],
        patience=CONFIG["patience"], close_mosaic=CONFIG["close_mosaic"],
        multi_scale=CONFIG["multi_scale"], amp=CONFIG["amp"],
        save_period=CONFIG["save_period"], plots=CONFIG["plots"],
        iou=CONFIG["iou"], box=CONFIG["box"], dfl=CONFIG["dfl"], cls_pw=CONFIG["cls_pw"],
        project="runs/detect",
        name=MODEL_NAME,
        exist_ok=True,
        verbose=True,
    )
    print("\n训练完成!")
    return results


def validate():
    best = Path("runs/detect") / MODEL_NAME / "weights" / "best.pt"
    if not best.exists():
        print(f"[ERROR] {best} 不存在")
        return
    model = YOLO(str(best))
    for split in ["val", "test"]:
        print(f"\n--- {split} ---")
        m = model.val(data=DATA_YAML, device=0, split=split)
        print(f"  mAP@0.5:      {m.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {m.box.map:.4f}")
        print(f"  Precision:    {m.box.mp:.4f}")
        print(f"  Recall:       {m.box.mr:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="仅检查环境/架构")
    ap.add_argument("--val", action="store_true", help="仅验证")
    args = ap.parse_args()

    if args.check:
        check_env()
    elif args.val:
        validate()
    else:
        check_env()
        train()
        validate()
