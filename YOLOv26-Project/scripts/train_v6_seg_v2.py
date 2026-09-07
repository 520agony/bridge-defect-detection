"""
final_6class seg 模型 v2 训练
============================
改进（吸取 v1 教训）：
  - batch 8→6（v1 在 43 轮 CUDA OOM）
  - overlap_mask=False（降低 mask 显存）
  - 已并入 2484 张真裂缝 polygon（v1 裂缝 mask mAP 只有 0.60 的根因是假 mask）
"""
import multiprocessing
multiprocessing.freeze_support()

from pathlib import Path
from ultralytics import YOLO

BASE = Path(r'd:\开始做项目\YOLOv26-Project')
WEIGHTS = BASE / 'weights' / 'yolov8s-seg.pt'
DATA = BASE / 'configs' / 'data_final_6class.yaml'
OUT = BASE / 'experiments' / 'v6_seg_v2'


def main():
    print(f"加载预训练: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    print(f"数据: {DATA}")
    print(f"输出: {OUT}")
    print("\n50 轮 v2 训练 ...")

    model.train(
        data=str(DATA),
        epochs=50,
        batch=6,            # 防 OOM
        imgsz=640,
        device=0,
        workers=0,           # 降到 0 省宿主内存（v2 报 HOST_ALLOCATION_FAILED）
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        dropout=0.1,
        patience=50,
        close_mosaic=15,
        overlap_mask=False,  # 降 mask 显存
        project=str(OUT.parent),
        name=OUT.name,
        exist_ok=True,
        plots=True,
        seed=0,
    )
    print("\n[DONE] v2 训练完成")


if __name__ == '__main__':
    main()
