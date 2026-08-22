"""
桥梁病害检测 - YOLOv8 基线训练脚本
Bridge Defect Detection with YOLOv8 Baseline
华东交通大学大创项目
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# ============================================================
# 配置参数
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/train -> src -> PROJECT_ROOT
if not PROJECT_ROOT.is_absolute():
    PROJECT_ROOT = Path.cwd()
DATASET_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# 训练参数
CONFIG = {
    # 模型
    "model": "yolov8n.pt",          # 基线模型 (nano, 6MB)
    "imgsz": 640,                    # 输入尺寸
    # 训练
    "epochs": 100,                   # 训练轮数
    "batch": 16,                     # 批大小 (8GB显存用16)
    "lr0": 0.01,                     # 初始学习率
    "lrf": 0.01,                     # 最终学习率因子
    "patience": 20,                  # 早停轮数
    "device": 0,                     # GPU设备 (None=auto)
    "workers": 4,                    # 数据加载线程
    # 数据增强
    "hsv_h": 0.015,                  # HSV色调增强
    "hsv_s": 0.7,                    # HSV饱和度增强
    "hsv_v": 0.4,                    # HSV明度增强
    "degrees": 10.0,                 # 旋转角度
    "translate": 0.1,                # 平移
    "scale": 0.5,                    # 缩放
    "shear": 2.0,                    # 剪切
    "fliplr": 0.5,                   # 水平翻转概率
    "mosaic": 1.0,                   # 马赛克增强
    "mixup": 0.1,                    # MixUp增强
}

def check_environment():
    """检查训练环境"""
    print("=" * 60)
    print("  环境检查")
    print("=" * 60)

    # Python
    print(f"Python:    {sys.version.split()[0]}")

    # PyTorch
    print(f"PyTorch:   {torch.__version__}")
    print(f"CUDA:       {torch.version.cuda}")

    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // 1024**2
        print(f"GPU:        {gpu_name}")
        print(f"VRAM:       {vram} MB")
    else:
        print("GPU:        NOT AVAILABLE! 请检查CUDA安装")
        sys.exit(1)

    # 数据集
    data_yaml = PROJECT_ROOT / "configs" / "data.yaml"
    if not data_yaml.exists():
        print(f"\n[ERROR] 数据集配置不存在: {data_yaml}")
        print("请先运行 prepare_dataset.py 准备数据")
        sys.exit(1)

    # 数据量检查
    import yaml
    with open(data_yaml, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    for split in ['train', 'val']:
        split_dir = DATASET_DIR / cfg[split]
        if split_dir.exists():
            # Count images
            imgs = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpeg"))
            print(f"{split}:        {len(imgs)} 张图片")
        else:
            print(f"{split}:        目录不存在!")

    print(f"\n[OK] 环境检查通过，可以开始训练")
    print("=" * 60)
    return True


def train():
    """执行训练"""
    data_yaml = str(PROJECT_ROOT / "configs" / "data.yaml")

    print("\n" + "=" * 60)
    print("  开始训练 YOLOv8 桥梁病害检测模型")
    print("=" * 60)
    print(f"\n模型:      {CONFIG['model']}")
    print(f"数据配置:  {data_yaml}")
    print(f"训练轮数:  {CONFIG['epochs']}")
    print(f"Batch:     {CONFIG['batch']}")
    print(f"图片尺寸:  {CONFIG['imgsz']}")
    print("")

    # 加载模型 (先找本地文件, 避免 GitHub 下载超时)
    model_path = PROJECT_ROOT / "yolov8n.pt"
    if not model_path.exists():
        model_path = Path("yolov8n.pt")
    print(f"加载模型: {model_path} (exists={model_path.exists()})")
    model = YOLO(str(model_path))

    # 训练
    results = model.train(
        data=data_yaml,
        epochs=CONFIG['epochs'],
        batch=CONFIG['batch'],
        imgsz=CONFIG['imgsz'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        patience=CONFIG['patience'],
        device=CONFIG['device'],
        workers=CONFIG['workers'],

        # 数据增强
        hsv_h=CONFIG['hsv_h'],
        hsv_s=CONFIG['hsv_s'],
        hsv_v=CONFIG['hsv_v'],
        degrees=CONFIG['degrees'],
        translate=CONFIG['translate'],
        scale=CONFIG['scale'],
        shear=CONFIG['shear'],
        fliplr=CONFIG['fliplr'],
        mosaic=CONFIG['mosaic'],
        mixup=CONFIG['mixup'],

        # 保存
        project=str(EXPERIMENTS_DIR),
        name="bridge_defect_baseline",
        exist_ok=True,

        # 日志
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  训练完成!")
    print("=" * 60)
    return results


def validate():
    """验证模型"""
    best_pt = EXPERIMENTS_DIR / "bridge_defect_baseline" / "weights" / "best.pt"

    if not best_pt.exists():
        print(f"[ERROR] 模型权重不存在: {best_pt}")
        return

    print("\n" + "=" * 60)
    print("  模型验证")
    print("=" * 60)

    model = YOLO(str(best_pt))
    metrics = model.val()

    print(f"\nmAP@0.5:   {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    return metrics


def export_models():
    """导出模型为 ONNX 和 TensorRT 格式（用于后续部署）"""
    best_pt = EXPERIMENTS_DIR / "bridge_defect_baseline" / "weights" / "best.pt"

    if not best_pt.exists():
        print(f"[ERROR] 模型权重不存在: {best_pt}")
        return

    model = YOLO(str(best_pt))

    # 导出 ONNX
    print("\n导出 ONNX...")
    model.export(format='onnx', imgsz=640, dynamic=True)
    print("ONNX 导出完成")

    # 导出 TensorRT (仅在有 GPU 时)
    if torch.cuda.is_available():
        print("\n导出 TensorRT...")
        try:
            model.export(format='engine', imgsz=640, half=True)
            print("TensorRT 导出完成")
        except Exception as e:
            print(f"TensorRT 导出失败: {e}")
            print("(这通常是正常的，TensorRT 在 Jetson 上导出更合适)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 桥梁病害检测训练")
    parser.add_argument("--check", action="store_true", help="仅检查环境")
    parser.add_argument("--val", action="store_true", help="仅验证已有模型")
    parser.add_argument("--export", action="store_true", help="导出模型")
    args = parser.parse_args()

    if args.check:
        check_environment()
    elif args.val:
        validate()
    elif args.export:
        export_models()
    else:
        # 完整训练流程
        check_environment()
        train()
        validate()
        export_models()
