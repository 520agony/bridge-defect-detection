"""
YOLOv26 Training Script v3 — Expanded Crack Data
==================================================
Training on: 16003 images, 8 classes
Model: yolov8s.pt (small, 11.1M params)
GPU: RTX 5060 Laptop 8GB

Strategy:
 — 300 epochs with warmup + cosine annealing
 — Strong crack-focused augmentation
 — batch=8, no copy_paste (VRAM safe)
"""

import os, sys
from pathlib import Path as P
from ultralytics import YOLO
import torch
import yaml

# ============================================================
# Project paths (absolute, resolved)
# ============================================================
PROJECT_ROOT = P(__file__).resolve().parent.parent.parent
assert PROJECT_ROOT.is_dir(), f"Bad project root: {PROJECT_ROOT}"
os.chdir(str(PROJECT_ROOT))

DATASET_DIR = (PROJECT_ROOT / "datasets").resolve()
DATA_YAML   = (PROJECT_ROOT / "configs" / "data.yaml").resolve()
print(f"Project:  {PROJECT_ROOT}")
print(f"Datasets: {DATASET_DIR}")
print(f"Data yaml:{DATA_YAML}")

# ============================================================
# Write data.yaml  —  use relative path to avoid Chinese encoding in ultralytics
# ============================================================
data_cfg = {
    # IMPORTANT: ultralytics resolves `path` via os.path.join — use raw string or relative
    "path": str(DATASET_DIR.as_posix()),  # posix: D:/.../datasets
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc": 8,
    "names": [
        "corrosion", "crack", "degraded_concrete", "concrete_hole",
        "moisture", "pavement_degrade", "shrinkage_crack", "sub_shrinkage_crack",
    ],
}

with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump(data_cfg, f, default_flow_style=False, allow_unicode=True)

# Verify what we wrote
with open(DATA_YAML, "r", encoding="utf-8") as f:
    print("--- data.yaml ---")
    print(f.read())
    print("-----------------")

# Verify the dataset directories exist
for s in ["train", "val", "test"]:
    d = DATASET_DIR / "images" / s
    lbl = DATASET_DIR / "labels" / s
    assert d.is_dir(), f"MISSING: {d}"
    assert lbl.is_dir(), f"MISSING: {lbl}"
    print(f"{s}: {len(list(d.iterdir()))} imgs / {len(list(lbl.iterdir()))} lbls  [OK]")

# ============================================================
# Training config
# ============================================================
MODEL_NAME = "yolov26_crack_v3"

CFG = dict(
    model="yolov8s.pt",
    imgsz=640,
    epochs=300,
    batch=8,         # RTX 5060 8GB — batch=8 safe with copy_paste
    device=0,
    workers=0,  # Windows needs 0 to avoid multiprocessing spawn crash
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.1,
    scale=0.6,
    shear=2.0,
    fliplr=0.5,
    perspective=0.0005,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.0,   # disabled — OOM risk with 8GB + 12527 images
    erasing=0.4,
    dropout=0.1,
    patience=50,
    close_mosaic=15,
    multi_scale=False,  # disabled — 8GB VRAM + batch=8
    amp=False,           # disabled AMP — avoid cuDNN crashes on RTX 5060
    iou=0.7,
    box=7.5,
    dfl=1.5,
    cls_pw=1.0,
)

# ============================================================
def env_ok():
    print(f"\nPython:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}  CUDA {torch.version.cuda}")
    assert torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
    print(f"GPU:     {gpu} ({vram} MiB)")
    print("[OK] Ready.\n")

# ============================================================
def train():
    model_name = CFG["model"]
    local = PROJECT_ROOT / model_name
    if local.exists():
        print(f"Loading local {model_name}")
    else:
        alt = PROJECT_ROOT / "yolov8n.pt"
        if alt.exists():
            model_name = "yolov8n.pt"
            CFG["model"] = model_name
            print("Using yolov8n.pt (yolov8s.pt not found locally)")
        else:
            print("No local model, trying download...")

    # Resume from last checkpoint if available
    last_pt = PROJECT_ROOT / "experiments" / MODEL_NAME / "weights" / "last.pt"
    if last_pt.exists():
        print(f"Resuming from {last_pt}")
        model = YOLO(str(last_pt))
        params = sum(p.numel() for p in model.model.parameters())
        print(f"Parameters: {params / 1e6:.1f} M")
        # Don't resume via ultralytics (CUDA context issue), load weights -> train fresh
        print(f"Loaded checkpoint, continuing training from epoch ~27+")
    else:
        model = YOLO(model_name)
        params = sum(p.numel() for p in model.model.parameters())
        print(f"Parameters: {params / 1e6:.1f} M")

    print(f"\n{'='*60}")
    print(f"  YOLOv26 v3 — Training Start")
    print(f"  Model: {model_name} ({params / 1e6:.1f}M)")
    print(f"  Data:  16003 train / 946 val / 947 test")
    print(f"  Epochs: {CFG['epochs']}  Batch: {CFG['batch']}")
    print(f"{'='*60}\n")

    results = model.train(
        data=str(DATA_YAML),
        epochs=CFG["epochs"],
        batch=CFG["batch"],
        imgsz=CFG["imgsz"],
        device=CFG["device"],
        workers=CFG["workers"],
        optimizer=CFG["optimizer"],
        lr0=CFG["lr0"],
        lrf=CFG["lrf"],
        momentum=CFG["momentum"],
        weight_decay=CFG["weight_decay"],
        warmup_epochs=CFG["warmup_epochs"],
        warmup_momentum=CFG["warmup_momentum"],
        warmup_bias_lr=CFG["warmup_bias_lr"],
        cos_lr=CFG["cos_lr"],
        hsv_h=CFG["hsv_h"],
        hsv_s=CFG["hsv_s"],
        hsv_v=CFG["hsv_v"],
        degrees=CFG["degrees"],
        translate=CFG["translate"],
        scale=CFG["scale"],
        shear=CFG["shear"],
        fliplr=CFG["fliplr"],
        perspective=CFG["perspective"],
        mosaic=CFG["mosaic"],
        mixup=CFG["mixup"],
        copy_paste=CFG["copy_paste"],
        erasing=CFG["erasing"],
        dropout=CFG["dropout"],
        patience=CFG["patience"],
        close_mosaic=CFG["close_mosaic"],
        multi_scale=CFG["multi_scale"],
        iou=CFG["iou"],
        box=CFG["box"],
        dfl=CFG["dfl"],
        cls_pw=CFG["cls_pw"],
        amp=CFG["amp"],
        project=str(PROJECT_ROOT / "experiments"),
        name=MODEL_NAME,
        exist_ok=True,
        resume=False,  # fresh train, not ultralytics resume
    )
    print(f"\n{'='*60}")
    print(f"  YOLOv26 v3 Training DONE!")
    print(f"{'='*60}")
    return results


def validate():
    best = PROJECT_ROOT / "experiments" / MODEL_NAME / "weights" / "best.pt"
    if not best.exists():
        print(f"[ERROR] {best} not found"); return
    model = YOLO(str(best))
    for s in ["val", "test"]:
        print(f"\n--- {s} ---")
        m = model.val(data=str(DATA_YAML), device=0, split=s)
        print(f"  mAP@0.5:      {m.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {m.box.map:.4f}")
        print(f"  Precision:    {m.box.mp:.4f}")
        print(f"  Recall:       {m.box.mr:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    env_ok()
    if args.val:
        validate()
    elif args.check:
        pass
    else:
        train()
        validate()
