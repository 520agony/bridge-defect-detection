"""
YOLOv26 改进版训练脚本 (v3 - yolov8s 版)
==========================================
改进策略:
  1. yolov8s backbone (11M param, 3x nano) + 更强的特征提取
  2. cos_lr + copy_paste + label_smoothing + dropout
  3. 更强的数据增强 (degrees/mixup/scale++)
  4. NO multi_scale — 保持稳定训练
  5. close_mosaic@15 — 最后15轮关闭mosaic稳定收敛
  6. AdamW optimizer

Target: mAP@0.5 > 0.40 on bridge defect 8-class detection
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(r'd:/开始做项目/YOLOv26-Project')
DATA_YAML = str(PROJECT_ROOT / 'configs' / 'data.yaml')

# ============================================================
# 配置
# ============================================================
def train():
    model_path = PROJECT_ROOT / 'yolov8s.pt'
    if model_path.exists() and model_path.stat().st_size > 20_000_000:
        model_name = str(model_path)
        print(f'Using local yolov8s.pt ({model_path.stat().st_size/1e6:.1f}MB)')
    else:
        print('yolov8s.pt not found/too small. Falling back to yolov8n.pt')
        model_name = 'yolov8n.pt'

    model = YOLO(model_name)
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f'Model: {model_name} ({total_params/1e6:.1f}M params)')
    print(f'Training: 100 epochs, AdamW, cos_lr, copy_paste, NO multi_scale')

    results = model.train(
        data=DATA_YAML,
        epochs=100,
        batch=16,
        imgsz=640,
        lr0=0.001, lrf=0.01,
        patience=30,
        device=0,
        workers=4,
        # Augmentations (stronger than baseline)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, translate=0.1, scale=0.6, shear=2.0,
        fliplr=0.5, mosaic=1.0, mixup=0.15,
        copy_paste=0.1, erasing=0.4,
        # Loss
        iou=0.7,
        # Training strategy
        cos_lr=True, close_mosaic=15,
        warmup_epochs=3, warmup_bias_lr=0.1,
        weight_decay=0.0005, label_smoothing=0.05, dropout=0.1,
        multi_scale=False,  # 关键: 不使用multi_scale
        # Optimizer
        optimizer='AdamW',
        # Save
        project=str(PROJECT_ROOT / 'experiments'),
        name='yolov26_s_v3',
        exist_ok=True,
        verbose=True,
    )

    # Validation on test set
    print('\n' + '=' * 60)
    print('  YOLOv26-s V3 测试集评估')
    print('=' * 60)

    best_pt = PROJECT_ROOT / 'experiments' / 'yolov26_s_v3' / 'weights' / 'best.pt'
    model = YOLO(str(best_pt))
    metrics = model.val(data=DATA_YAML, device=0, split='test')

    print(f'\nmAP@0.5:      {metrics.box.map50:.4f}')
    print(f'mAP@0.5:0.95: {metrics.box.map:.4f}')
    print(f'Precision:    {metrics.box.mp:.4f}')
    print(f'Recall:       {metrics.box.mr:.4f}')

    # Export ONNX
    print('\nExporting ONNX...')
    model.export(format='onnx', imgsz=640, dynamic=True)
    print('Done!')

    return results


if __name__ == '__main__':
    train()
