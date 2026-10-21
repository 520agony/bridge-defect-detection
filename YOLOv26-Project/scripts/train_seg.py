"""
train_seg.py - 5 类 seg 模型高效训练（v2_clean 阶段）
====================================================
吸取 v1/v2/v3 全部教训的最终训练方案：

  1. **imgsz 960**（v3 已证对小目标好 1 倍显存，5.4G 数据可接受）
  2. **batch 6 + overlap_mask=False**（显存保护，防 OOM）
  3. **workers 0**（宿主内存保护，16G 内存的电脑）
  4. **patience 20**（更激进早停，省时间）
  5. **close_mosaic 10**（提前关 mosaic，少破坏小目标）
  6. **cos_lr + warmup 3**（训练稳定）
  7. **--resume 自动从 last.pt 续**（断电/关机不丢进度）

用法：
  python scripts/train_seg.py             # 首次训练
  python scripts/train_seg.py --resume    # 续训
  python scripts/train_seg.py --fresh     # 覆盖重训
"""
import argparse
import multiprocessing
multiprocessing.freeze_support()

from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent
WEIGHTS = BASE / 'weights' / 'yolov8s-seg.pt'
DATA = BASE / 'configs' / 'data_final_5class_clean.yaml'  # 清洗+并入后数据（旧 dirty 用 data_final_5class.yaml，已废弃）
OUT = BASE / 'experiments' / 'seg_v2_clean'

EPOCHS = 80
BATCH = 6
IMGSZ = 960
LR0 = 0.001
PATIENCE = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true', help='从 last.pt 续训')
    ap.add_argument('--fresh', action='store_true', help='完全重训')
    ap.add_argument('--epochs', type=int, default=EPOCHS, help=f'总轮数（默认 {EPOCHS}）')
    ap.add_argument('--batch', type=int, default=BATCH, help=f'batch（默认 {BATCH}）')
    ap.add_argument('--imgsz', type=int, default=IMGSZ, help=f'训练尺寸（默认 {IMGSZ}）')
    ap.add_argument('--name', default='seg_v2_clean', help='实验名')
    args = ap.parse_args()

    last = OUT / 'weights' / 'last.pt'
    if args.fresh and last.exists():
        print(f'[INFO] --fresh: 删 {last}')
        last.unlink()
    if not args.fresh and last.exists():
        print(f'[INFO] 续训: 从 {last} 恢复（epochs={args.epochs}, imgsz={args.imgsz}）')
        m = YOLO(str(last))
        m.train(resume=True, project=str(OUT.parent), name=args.name, exist_ok=True)
        return

    if not WEIGHTS.exists():
        print(f'[WARN] 预训练 {WEIGHTS} 不在')
        print('       你需要手动放到 weights/yolov8s-seg.pt（不是 yolo26n）')
        return
    print(f'[INFO] 新训: {WEIGHTS} -> {OUT}')
    print(f'       epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} patience={PATIENCE}')
    m = YOLO(str(WEIGHTS))
    m.train(
        data=str(DATA),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        workers=0,
        optimizer='AdamW',
        lr0=LR0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        dropout=0.1,
        patience=PATIENCE,
        close_mosaic=10,
        overlap_mask=False,
        project=str(OUT.parent),
        name=args.name,
        exist_ok=True,
        plots=True,
        seed=0,
    )


if __name__ == '__main__':
    main()
