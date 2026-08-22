"""
YOLOv26 v4 — Stable Training (resume from best checkpoint)
Uses best.pt from v3 (~epoch 26, mAP@0.5=0.1086) to continue training
Batch=8, AMP off, workers=0, conservative augmentation
"""

import os, sys, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(PROJECT_ROOT))

import torch
gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False

from ultralytics import YOLO
import yaml

# === Data YAML (clean) ===
DATASET_DIR = (PROJECT_ROOT / "datasets").resolve()
DATA_YAML   = PROJECT_ROOT / "configs" / "data.yaml"

with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump({
        "path": DATASET_DIR.as_posix(),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc": 8,
        "names": ["corrosion","crack","degraded_concrete","concrete_hole",
                   "moisture","pavement_degrade","shrinkage_crack","sub_shrinkage_crack"]
    }, f, default_flow_style=False, allow_unicode=True)

MODEL_NAME = "yolov26_crack_v4"
CHECKPOINT = PROJECT_ROOT / "experiments" / "yolov26_crack_v3" / "weights" / "best.pt"

print(f"Python:  {sys.version.split()[0]}")
print(f"Torch:   {torch.__version__}  CUDA {torch.version.cuda}")
gpu = torch.cuda.get_device_name(0); vram = torch.cuda.get_device_properties(0).total_memory//1024**2
print(f"GPU:     {gpu}  ({vram} MiB)")
print(f"Checkpoint: {CHECKPOINT} (exists={CHECKPOINT.exists()})")

# Count data
for s in ["train","val","test"]:
    n = len(list((DATASET_DIR/"images"/s).iterdir()))
    print(f"  {s}: {n} imgs")

# Load checkpoint
model = YOLO(str(CHECKPOINT)) if CHECKPOINT.exists() else YOLO("yolov8n.pt")
params = sum(p.numel() for p in model.model.parameters())
print(f"\nModel: {params/1e6:.1f}M params. Training 300 epochs, batch=8, AMP=off")

# === Train ===
results = model.train(
    data=str(DATA_YAML),
    epochs=300,
    batch=8,
    imgsz=640,
    device=0,
    workers=0,

    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=2,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,

    hsv_h=0.01, hsv_s=0.5, hsv_v=0.3,
    degrees=5.0, translate=0.1, scale=0.3,
    shear=0.0, fliplr=0.5, perspective=0.0,
    mosaic=0.5, mixup=0.0, copy_paste=0.0, erasing=0.0,
    dropout=0.0,

    patience=80,
    close_mosaic=10,
    multi_scale=False,
    amp=False,

    iou=0.7, box=7.5, dfl=1.5, cls_pw=1.0,

    project=str(PROJECT_ROOT/"experiments"),
    name=MODEL_NAME,
    exist_ok=True,
    resume=False,
    verbose=True,
)

print("\n=== TRAINING DONE ===")

# === Validate ===
best = PROJECT_ROOT/"experiments"/MODEL_NAME/"weights"/"best.pt"
if best.exists():
    model = YOLO(str(best))
    for s in ["val","test"]:
        m = model.val(data=str(DATA_YAML), device=0, split=s)
        print(f"{s}: mAP@0.5={m.box.map50:.4f}  mAP@0.5:0.95={m.box.map:.4f}  P={m.box.mp:.4f}  R={m.box.mr:.4f}")
