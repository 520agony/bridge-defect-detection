"""
推理速度基准测试（FPS / 延迟）
================================
对 best.pt 分别测 GPU(CUDA fp32)、CPU(fp32)、以及（可选）导出的 ONNX 模型，
输出平均延迟和 FPS，用于申报书「≥30 FPS」指标与部署评估。

用法:
    ./.venv/Scripts/python.exe src/eval/benchmark.py \
        --weights experiments/yolov26_baseline_v2/weights/best.pt

参数:
    --weights  模型权重路径
    --imgsz    输入尺寸（默认 640）
    --n        正式计时迭代次数（默认 100，GPU 用；CPU 建议 --n 30）
    --warmup   预热次数（默认 10）
    --onnx     可选，onnx 文件路径（测 onnxruntime）
"""

import argparse
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def bench_predict(model, device, img, imgsz, n, warmup):
    """用 model.predict 端到端计时（含预处理/推理/NMS），返回 (延迟ms, FPS)"""
    for _ in range(warmup):
        model.predict(img, imgsz=imgsz, device=device, verbose=False)

    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
    dt = (time.perf_counter() - t0) / n
    return dt * 1000.0, 1.0 / dt


def bench_onnx(session, img, n, warmup):
    """用 onnxruntime 计时（纯推理，不含 NMS），返回 (延迟ms, FPS)"""
    import onnxruntime as ort

    # letterbox 到 640 的 uint8 输入
    x = np.ascontiguousarray(img[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
    iname = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {iname: x})
    t0 = time.perf_counter()
    for _ in range(n):
        session.run(None, {iname: x})
    dt = (time.perf_counter() - t0) / n
    return dt * 1000.0, 1.0 / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--onnx", default=None)
    args = ap.parse_args()

    img = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    model = YOLO(args.weights)

    print("=" * 56)
    print("  推理速度基准（imgsz=%d, 正式计时 %d 次）" % (args.imgsz, args.n))
    print("=" * 56)

    # GPU
    try:
        ms, fps = bench_predict(model, 0, img, args.imgsz, args.n, args.warmup)
        print(f"GPU (CUDA fp32)   : {ms:6.2f} ms   {fps:7.1f} FPS")
    except Exception as e:
        print(f"GPU (CUDA fp32)   : 失败 {e}")

    # CPU
    try:
        n_cpu = max(args.n // 3, 10)
        ms, fps = bench_predict(model, "cpu", img, args.imgsz, n_cpu, args.warmup // 2)
        print(f"CPU (fp32)        : {ms:6.2f} ms   {fps:7.1f} FPS")
    except Exception as e:
        print(f"CPU (fp32)        : 失败 {e}")

    # ONNX（可选）
    if args.onnx and Path(args.onnx).exists():
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print(f"\nonnxruntime providers: {providers}")
            for ep in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
                if ep in providers:
                    sess = ort.InferenceSession(args.onnx, providers=[ep])
                    ms, fps = bench_onnx(sess, img, args.n, args.warmup)
                    print(f"ONNX ({ep:22s}): {ms:6.2f} ms   {fps:7.1f} FPS  (纯推理，不含 NMS)")
        except Exception as e:
            print(f"ONNX 基准失败: {e}")

    print("=" * 56)
    print("注：此为本机 RTX 5060 / CPU 的数值；申报书 30FPS 针对")
    print("    Jetson + TensorRT + INT8 边缘部署，将在边缘阶段另行测试。")


if __name__ == "__main__":
    main()
