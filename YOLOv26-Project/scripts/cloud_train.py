"""
cloud_train.py - 云服务器训练一键脚本（4090 24G 预设）
=====================================================
三阶段，对应论文表 1/2/3：
  --mode s        ① yolo26s-seg 新基线（对照本地 v8s）
  --mode m        ② yolo26m-seg 教师（冲精度上限）
  --mode distill  ③ m -> s 蒸馏（部署版，核心创新）

用法（项目根目录）：
  python scripts/cloud_train.py --mode s --smoke      # 冒烟：1 epoch 验证链路
  python scripts/cloud_train.py --mode s              # 正式
  python scripts/cloud_train.py --mode m
  python scripts/cloud_train.py --mode distill --teacher experiments/cloud_m/weights/best.pt
  python scripts/cloud_train.py --mode s --resume     # 断点续训
"""
import argparse
import multiprocessing

multiprocessing.freeze_support()

from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "configs" / "data_cloud_5class.yaml"

MODES = {
    # ① s 基线：batch 32 + imgsz 1024 + copy-paste（小目标）
    "s": dict(weights=BASE / "weights" / "yolo26s-seg.pt",
              name="cloud_s", epochs=200, batch=32, imgsz=1024,
              lr0=0.001, patience=30, extra=dict(copy_paste=0.3)),
    # ② m 教师：batch 16（m 更大），其余同 s
    "m": dict(weights=BASE / "weights" / "yolo26m-seg.pt",
              name="cloud_m", epochs=300, batch=16, imgsz=1024,
              lr0=0.001, patience=40, extra=dict(copy_paste=0.3)),
    # ③ 蒸馏：s 学生 + m 教师，dis 为蒸馏损失权重（官方默认 6.0）
    "distill": dict(weights=BASE / "weights" / "yolo26s-seg.pt",
                    name="cloud_s_distill", epochs=200, batch=32, imgsz=1024,
                    lr0=0.001, patience=30,
                    extra=dict(copy_paste=0.3, dis=6.0)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--smoke", action="store_true", help="只跑 1 epoch 验证链路")
    ap.add_argument("--resume", action="store_true", help="从 last.pt 续训")
    ap.add_argument("--teacher", default="", help="蒸馏教师权重（默认用云 m 的 best.pt）")
    ap.add_argument("--epochs", type=int, default=0, help="覆盖默认轮数")
    ap.add_argument("--batch", type=int, default=0, help="覆盖默认 batch")
    ap.add_argument("--data", default="", help="覆盖数据配置（冒烟小数据集用）")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers（Kaggle改2）")
    args = ap.parse_args()

    data = args.data or str(DATA)
    cfg = MODES[args.mode]
    out = BASE / "experiments" / cfg["name"]
    epochs = args.epochs or (1 if args.smoke else cfg["epochs"])
    batch = args.batch or cfg["batch"]

    last = out / "weights" / "last.pt"
    if args.resume and last.exists():
        print(f"[INFO] 续训: {last}")
        YOLO(str(last)).train(resume=True, project=str(out.parent),
                              name=cfg["name"], exist_ok=True)
        return

    distill_model = None
    if args.mode == "distill":
        distill_model = args.teacher or str(BASE / "experiments" / "cloud_m" / "weights" / "best.pt")
        if not Path(distill_model).exists():
            print(f"[ERROR] 教师权重不存在: {distill_model}")
            print("        先跑 --mode m，或用 --teacher 指定路径")
            return
        print(f"[INFO] 蒸馏教师: {distill_model}")

    print(f"[INFO] {args.mode}: {cfg['weights'].name} -> {out}")
    print(f"       epochs={epochs} batch={batch} imgsz={cfg['imgsz']} "
          f"patience={cfg['patience']}" + (" [SMOKE]" if args.smoke else ""))
    m = YOLO(str(cfg["weights"]))
    m.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=cfg["imgsz"],
        device=0,
        workers=args.workers,
        optimizer="AdamW",
        lr0=cfg["lr0"],
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        dropout=0.1,
        patience=cfg["patience"],
        close_mosaic=15,
        overlap_mask=True,   # 云上显存够，开 overlap 保 mask 精度
        distill_model=distill_model,
        project=str(out.parent),
        name=cfg["name"],
        exist_ok=True,
        plots=True,
        seed=0,
        **cfg["extra"],
    )


if __name__ == "__main__":
    main()
