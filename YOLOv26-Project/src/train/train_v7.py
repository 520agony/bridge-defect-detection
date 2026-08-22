"""
YOLOv26 v7 — 官方 6 类桥梁病害检测
=====================================
数据：merged_6class（官方 6 类 + 先前数据映射补充），共 29808 张
类别：Crack / Breakage / Comb / Hole / Reinforcement / Seepage
模型：yolov8s.pt 预训练
"""

import torch
import gc
from pathlib import Path
import yaml

gc.collect()
torch.cuda.empty_cache()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import os
os.chdir(str(PROJECT_ROOT))

from ultralytics import YOLO

DATA_YAML = "configs/data_6class.yaml"
MODEL_NAME = "yolov26_v7_6class"

# 预训练权重
pretrained = PROJECT_ROOT / "yolov8s.pt"

print(f"数据配置: {DATA_YAML}")
print(f"预训练权重: {pretrained.name} (exists={pretrained.exists()})")

model = YOLO(str(pretrained))

model.train(
    data=DATA_YAML,
    epochs=300,
    batch=4,          # 内存受限，batch=4 安全
    imgsz=640,
    device=0,
    workers=0,        # Windows 必须为 0
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    hsv_h=0.01, hsv_s=0.5, hsv_v=0.3,
    degrees=5.0, translate=0.1, scale=0.3,
    shear=0.0, fliplr=0.5, perspective=0.0,
    mosaic=0.3, mixup=0.0, copy_paste=0.0, erasing=0.0,
    dropout=0.0,
    patience=80, close_mosaic=10,
    multi_scale=False, amp=False,
    save_period=10,   # 内存优化
    plots=False,      # 关闭预览图
    iou=0.7, box=7.5, dfl=1.5, cls_pw=1.0,
    project="runs/detect",
    name=MODEL_NAME,
    exist_ok=True,
    resume=False,
    verbose=True,
)
print("DONE!")
torch.cuda.empty_cache()
