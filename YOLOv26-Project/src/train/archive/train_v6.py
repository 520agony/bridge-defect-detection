"""v6 微调：加载 v5 权重 + 渗水补充集，微调 60 轮，对比 moisture AP"""
import torch
import gc
import time
from pathlib import Path
import yaml

gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 关键修复：monkey-patch torch.save 的封装，让 ValueError(I/O closed file)
# 也能重试。这是 Windows 杀毒软件/系统扫描文件导致的间歇性文件锁，
# ultralytics 原生只重试 RuntimeError，导致遇到 ValueError 直接崩溃。
# ============================================================
import ultralytics.utils.patches as _patches

_original_torch_save = _patches.torch_save

def _robust_torch_save(*args, **kwargs):
    for i in range(5):  # 最多5次尝试
        try:
            return _original_torch_save(*args, **kwargs)
        except (RuntimeError, ValueError, OSError) as e:
            if i == 4:
                raise
            time.sleep((2 ** i) / 2)  # 0.5s, 1s, 2s, 4s 退避

_patches.torch_save = _robust_torch_save
print("[修复] 已启用 torch.save 重试机制（含 ValueError/OSError）")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import os
os.chdir(str(PROJECT_ROOT))

from ultralytics import YOLO

DATASET_DIR = PROJECT_ROOT / "datasets"
DATA_YAML = PROJECT_ROOT / "configs" / "data.yaml"

with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump({
        "path": DATASET_DIR.as_posix(),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": 8,
        "names": ["corrosion", "crack", "degraded_concrete", "concrete_hole",
                   "moisture", "pavement_degrade", "shrinkage_crack", "sub_shrinkage_crack"]
    }, f, default_flow_style=False, allow_unicode=True)

ckpt = PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "yolov26_crack_v5" / "weights" / "best.pt"
print(f"加载 v5 权重: {ckpt} (exists={ckpt.exists()})")

model = YOLO(str(ckpt))

model.train(
    data=str(DATA_YAML),
    epochs=60,
    batch=4,     # 降到4，进一步降低内存峰值（系统内存紧张）
    imgsz=640,
    device=0,
    workers=0,
    optimizer="AdamW", lr0=0.0001, lrf=0.01,
    momentum=0.937, weight_decay=0.0005,
    warmup_epochs=2, warmup_momentum=0.8, warmup_bias_lr=0.1,
    cos_lr=True,
    hsv_h=0.01, hsv_s=0.5, hsv_v=0.3,
    degrees=5.0, translate=0.1, scale=0.3,
    shear=0.0, fliplr=0.5, perspective=0.0,
    mosaic=0.3, mixup=0.0, copy_paste=0.0, erasing=0.0,
    dropout=0.0,
    patience=30, close_mosaic=10,
    multi_scale=False, amp=False,
    # 内存优化：每10轮才全量保存一次 checkpoint，减少 BytesIO 内存峰值
    save_period=10,
    plots=False,  # 关闭训练过程预览图生成，省内存
    iou=0.7, box=7.5, dfl=1.5, cls_pw=1.0,
    project="runs/detect",
    name="yolov26_crack_v6",
    exist_ok=True,
    resume=False,
    verbose=True,
)
print("DONE!")
torch.cuda.empty_cache()
