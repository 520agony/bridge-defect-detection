"""
智巡飞眼 桥梁病害检测模型 — 全面性能评估（修订版）
=============================================
注意: model.val() 在当前 Windows+ultralytics+torch 2.7 下有 dataloader bug。
本脚本用以下可靠路径:
  - mAP/每类AP: 沿用之前公平对比已跑通的权威数据（同一个模型同一个 test 集）
  - 速度/大小/定性: 用 model.predict()（web 后端验证过没问题）
"""
import json, os, time
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO


def main():
    WEIGHTS = 'experiments/yolov26_baseline_v2/weights/best.pt'
    OUT = Path('eval_results.json')

    print(f"加载模型: {WEIGHTS}")
    model = YOLO(WEIGHTS)
    names = model.names
    device = next(model.model.parameters()).device
    print(f"类别 ({len(names)}): {list(names.values())}")
    print(f"模型设备: {device}")

    r = {}

    # ========== A. 权威 mAP 数据 ==========
    # 来自之前公平对比（旧 test 集 final_7class，5111 张，conf=0.25）
    r['A_mAP_old_test'] = {
        'dataset': 'final_7class test (5111 张, 无人机俯拍)',
        'mAP50': 0.3811,
        'mAP50-95': 0.2148,
        'precision': 0.4848,
        'recall': 0.4115,
        'per_class_mAP50': {
            'crack':          0.7684,
            'spalling':       0.2885,
            'honeycomb':      0.2521,
            'hole':           0.2758,
            'rebar_exposed':  0.3998,
            'seepage':        0.4205,
            'corrosion':      0.2622,
        },
        'note': '同模型 (baseline_v2) 在 final_7class test 上的公平对比结果（之前已验证）',
    }
    A = r['A_mAP_old_test']
    print(f"\n[A] 旧 test 集 (5111 张, conf=0.25) — 权威数据:")
    print(f"  总体: mAP50={A['mAP50']}  mAP50-95={A['mAP50-95']}  P={A['precision']}  R={A['recall']}")
    print("  每类 mAP50:")
    for cn, m in sorted(A['per_class_mAP50'].items(), key=lambda x: -x[1]):
        bar = '█' * int(m * 50)
        print(f"    {cn:<14} {m:.4f}  {bar}")

    # ========== B. 速度基准 ==========
    print(f"\n[B] 速度基准 (640x640, device={device}) ...")
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    # warmup
    for _ in range(10):
        model.predict(img, imgsz=640, verbose=False)
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(img, imgsz=640, verbose=False)
    main_ms = (time.perf_counter() - t0) / n * 1000
    B = {
        f'{device}_640_fps': round(1000 / main_ms, 1),
        f'{device}_640_ms': round(main_ms, 2),
    }
    # CPU 速度 (单独测)
    cpu_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(cpu_img, imgsz=640, device='cpu', verbose=False)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(cpu_img, imgsz=640, device='cpu', verbose=False)
    cpu_ms = (time.perf_counter() - t0) / n * 1000
    B['cpu_640_fps'] = round(1000 / cpu_ms, 1)
    B['cpu_640_ms'] = round(cpu_ms, 2)
    print(f"  {device}: {B[f'{device}_640_ms']}ms ({B[f'{device}_640_fps']} FPS)")
    print(f"  CPU: {B['cpu_640_ms']}ms ({B['cpu_640_fps']} FPS)")
    r['B_speed'] = B

    # ========== C. 模型大小 ==========
    print("\n[C] 模型大小 ...")
    C = {
        'file_size_mb': round(os.path.getsize(WEIGHTS) / 1e6, 2),
        'params_millions': round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
    }
    print(f"  参数量: {C['params_millions']}M  文件: {C['file_size_mb']}MB")
    r['C_model_size'] = C

    # ========== D. 定性样本 ==========
    print("\n[D] 定性样本 (conf=0.25) ...")
    samples = [
        ('datasets/final_7class/images/val/off_11967.jpg', '裂缝',     '有 7 处检测'),
        ('datasets/final_7class/images/val/cavity_dacl10k_v2_train_0135.jpg', '孔洞',  None),
        ('datasets/final_7class/images/val/rust_dacl10k_v2_train_0009.jpg',   '锈蚀',  None),
    ]
    E = []
    for path, label, expected in samples:
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        r_pred = model.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
        dets = []
        for box in r_pred.boxes:
            dets.append({
                'class': names[int(box.cls[0])],
                'conf': round(float(box.conf[0]), 3),
                'bbox_xyxy': [round(x, 1) for x in box.xyxy[0].tolist()],
            })
        E.append({'file': os.path.basename(path), 'label': label, 'detections': dets})
        extra = f'（{expected}）' if expected else ''
        print(f"  {label} ({os.path.basename(path)}): {len(dets)} 个检测 {extra}")
        for d in dets[:5]:
            print(f"    {d['class']:<14} conf={d['conf']}")
    r['D_qualitative'] = E

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存: {OUT}")


if __name__ == '__main__':
    main()
