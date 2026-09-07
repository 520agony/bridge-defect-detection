"""
train_seg.py - seg 模型高效训练脚本
====================================
设计要点（吸取 v1/v2/v3 教训）：
  1. 显存保护：batch 6 + overlap_mask=False
  2. 宿主内存保护：workers 0
  3. 断点恢复：每次启动自动从 last.pt 续训（防止断电/关机丢失进度）
  4. 早停：patience 30
  5. 训练日志：results.csv 实时写入
  6. 后台安全：退出后进程不残留

用法：
  # 首次训练
  python scripts/train_seg.py

  # 从断点续训（断电、关机、停训后直接跑）
  python scripts/train_seg.py --resume

  # 完全重训（删 weights 后）
  python scripts/train_seg.py --fresh
"""
import argparse
import multiprocessing
multiprocessing.freeze_support()

from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
WEIGHTS = BASE / 'weights' / 'yolov8s-seg.pt'
DATA = BASE / 'configs' / 'data_final_6class_v3.yaml'
OUT = BASE / 'experiments' / 'seg_v2_clean'

# 训练超参
EPOCHS = 100
BATCH = 6
IMGSZ = 640
LR0 = 0.001
PATIENCE = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true', help='从 last.pt 续训（默认行为）')
    ap.add_argument('--fresh', action='store_true', help='完全重训，覆盖现有权重')
    ap.add_argument('--epochs', type=int, default=EPOCHS, help=f'总轮数（默认 {EPOCHS}）')
    ap.add_argument('--batch', type=int, default=BATCH, help=f'batch（默认 {BATCH}）')
    ap.add_argument('--name', default='seg_v2_clean', help='实验名')
    args = ap.parse_args()

    last = OUT / 'weights' / 'last.pt'
    if args.fresh and last.exists():
        print(f'[INFO] --fresh: 删 {last}')
        last.unlink()
    if not args.fresh and last.exists():
        print(f'[INFO] 续训: 从 {last} 恢复')
        m = YOLO(str(last))
        m.train(resume=True, project=str(OUT.parent), name=args.name, exist_ok=True)
        return

    # 新训
    if not WEIGHTS.exists():
        print(f'[WARN] 预训练 {WEIGHTS} 不在')
        print('       你需要手动放到 weights/yolov8s-seg.pt（不是 yolo26n）')
        print('       或传任意 yolov8s-seg.pt 的绝对路径给 model')
        return
    print(f'[INFO] 新训: 预训练 {WEIGHTS} -> 输出 {OUT} (epochs={args.epochs}, batch={args.batch})')
    m = YOLO(str(WEIGHTS))
    m.train(
        data=str(DATA),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=IMGSZ,
        device=0,
        workers=0,
        optimizer='AdamW',
        lr0=LR0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        dropout=0.1,
        patience=PATIENCE,
        close_mosaic=15,
        overlap_mask=False,
        project=str(OUT.parent),
        name=args.name,
        exist_ok=True,
        plots=True,
        seed=0,
    )


if __name__ == '__main__':
    main()
