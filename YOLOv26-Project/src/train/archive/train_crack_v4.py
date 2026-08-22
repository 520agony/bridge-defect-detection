"""
YOLOv26 Training Script v4 — Conservative Config for RTX 5060 Stability
=======================================================================
Minimal augmentation, no AMP, batch=6, 300 epochs
Loads the best previous checkpoint to continue from epoch ~26
"""

import os, sys, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
assert PROJECT_ROOT.is_dir()
os.chdir(str(PROJECT_ROOT))

# Clean slate
gc.collect()

import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False

from ultralytics import YOLO
import yaml

# ---- Data config ----
DATASET_DIR = (PROJECT_ROOT / "datasets").resolve()
DATA_YAML = (PROJECT_ROOT / "configs" / "data.yaml").resolve()

data_cfg = {
    "path": DATASET_DIR.as_posix(),
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

# ---- Training Config (conservative) ----
MODEL_NAME = "yolov26_crack_v4"

CFG = dict(
    # Using best checkpoint from v3 run as starting point
    model="experiments/yolov26_crack_v3/weights/best.pt",
    imgsz=640,
    epochs=300,
    batch=6,
    device=0,
    workers=0,
    # Mild optimizer
    optimizer="AdamW",
    lr0=0.0005,          # lower LR when starting from checkpoint
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=2,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    # Light augmentation
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=5.0,
    translate=0.1,
    scale=0.3,
    shear=0.0,
    fliplr=0.5,
    perspective=0.0,
    mosaic=0.5,          # reduced mosaic
    mixup=0.0,           # no mixup
    copy_paste=0.0,      # no copy_paste
    erasing=0.0,         # no erasing
    # Regularization
    dropout=0.0,
    # Training control
    patience=80,
    close_mosaic=10,
    multi_scale=False,
    amp=False,           # DISABLED — RTX 5060 cuDNN crash
    # Loss
    iou=0.7,
    box=7.5,
    dfl=1.5,
    cls_pw=1.0,
)

# ---- Main ----
def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} CUDA {torch.version.cuda}")
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory // 1024**2
    print(f"GPU: {gpu} ({vram} MiB)")

    # Verify dataset
    for split in ["train", "val", "test"]:
        d = DATASET_DIR / "images" / split
        l = DATASET_DIR / "labels" / split
        assert d.is_dir() and l.is_dir(), f"Missing {split}"
        print(f"  {split}: {len(list(d.iterdir()))} imgs / {len(list(l.iterdir()))} lbls")

    # Load model
    model_path = CFG["model"]
    if Path(model_path).exists():
        print(f"\nLoading checkpoint: {model_path}")
        model = YOLO(model_path)
    else:
        print(f"\nCheckpoint not found, using yolov8n.pt")
        model = YOLO("yolov8n.pt")

    params = sum(p.numel() for p in model.model.parameters())
    print(f"Parameters: {params/1e6:.1f} M")

    print(f"\n{'='*55}")
    print(f"  YOLOv26 v4 Training — Conservative Config")
    print(f"  Epochs: {CFG['epochs']}  Batch: {CFG['batch']}  AMP: OFF")
    print(f"  Aug: light mosaic only, no mixup/copy_paste")
    print(f"{'='*55}\n")

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
        amp=CFG["amp"],
        iou=CFG["iou"],
        box=CFG["box"],
        dfl=CFG["dfl"],
        cls_pw=CFG["cls_pw"],
        project=str(PROJECT_ROOT / "experiments"),
        name=MODEL_NAME,
        exist_ok=True,
        resume=False,
        verbose=True,
    )

    print(f"\n{'='*55}")
    print(f"  Training DONE!")
    print(f"{'='*55}")
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
    args = ap.parse_args()
    if args.val:
        validate()
    else:
        main()
