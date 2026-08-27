"""
Grad-CAM 热力图可视化脚本（手写，不依赖 pytorch-grad-cam）
============================================================
对 best.pt 做 Grad-CAM：钩住目标层特征 + 梯度，对目标类别的最高分检测反向传播，
生成热力图叠加在原图上，直观展示模型关注哪里。

用法:
    # 单张 / 目录（默认对全局最高分检测做 CAM）
    ./.venv/Scripts/python.exe src/eval/gradcam.py \
        --weights experiments/yolov26_baseline_v2/weights/best.pt \
        --source datasets/final_7class/images/val \
        --out experiments/yolov26_baseline_v2/gradcam --limit 7

    # 指定类别（只关注该类最高分）
    ./.venv/Scripts/python.exe src/eval/gradcam.py --weights ... --source xxx.jpg --class crack

    # 每类各挑一张（自动从 labels 里找包含该类的图，7 类 -> 7 张）
    ./.venv/Scripts/python.exe src/eval/gradcam.py --weights ... \
        --source datasets/final_7class/images/val \
        --labels datasets/final_7class/labels/val --out ... --per-class

参数:
    --weights    模型权重路径
    --source     图片、目录
    --labels     YOLO 标注目录（--per-class 时需要，用于按类挑图）
    --out        输出目录
    --class      指定类别名或索引（默认对全局最高分）
    --per-class  每类各生成一张热力图
    --layer      目标层索引（默认 -2 = 最后一个 C2f，位于 Detect 之前）
    --limit      目录模式最多处理张数（默认 10）
    --conf       置信度阈值（保留参数，当前未用于筛选）
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


def preprocess(img_path, imgsz=640):
    """读取并预处理为 [1,3,640,640] 张量（BGR -> RGB，HWC -> CHW，归一化）"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0), img


def grad_cam(model, x, target_layer_idx, class_id=None):
    """对单张输入 x 计算目标层的 Grad-CAM 热力图 (H, W)

    class_id=None 时对全局最高分检测反向传播；否则对指定类别最高分反向传播。
    返回 (cam, score)，score 为该次反向传播对应的类别分数（用于诊断）。
    """
    detect = model.model  # DetectionModel
    detect.eval()

    layer = detect.model[target_layer_idx]  # 支持负索引

    store = {}
    def fwd_hook(m, i, o):
        store["act"] = o
    def bwd_hook(m, gi, go):
        store["grad"] = go[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    try:
        x = x.clone().requires_grad_(True)
        out = detect(x)  # eval 模式返回 (预测张量, 特征 dict)

        pred = out[0] if isinstance(out, (tuple, list)) else out
        pred = pred[0]  # [1, 4+nc, N] -> [4+nc, N] 去掉 batch 维

        nc = len(model.names)
        class_scores = pred[4:4 + nc]  # [nc, N]

        # 目标：全局最高分，或指定类别最高分
        if class_id is None:
            target = class_scores.max()  # 整个 [nc, N] 的标量最大值
        else:
            target = class_scores[class_id].max()

        detect.zero_grad()
        target.backward()

        act = store["act"]        # [1, C, H, W]
        grad = store["grad"]      # [1, C, H, W]

        weights = grad.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(640, 640), mode="bilinear", align_corners=False)
        cam = cam.squeeze()  # [640, 640]
        cam = cam.detach().cpu().numpy()
        # 分位数归一化（99 分位代替最大值）：避免个别极值像素把热区压成一点（视觉上全蓝）
        p99 = np.percentile(cam, 99)
        cam = (cam - cam.min()) / (p99 - cam.min() + 1e-8)
        cam = np.clip(cam, 0, 1)
        cam = cv2.GaussianBlur(cam, (0, 0), 3)  # 高斯平滑，让热点区域更自然
        return cam, float(target.detach().cpu())
    finally:
        h1.remove()
        h2.remove()


def overlay_heatmap(img_rgb, cam, alpha=0.5):
    """把热力图叠加到原图（RGB）上，返回 BGR 图像"""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (1 - alpha) * img_rgb.astype(np.float32) + alpha * heatmap.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


def resolve_class_id(model, arg):
    """把 --class 参数解析为类别索引；None/空 -> None"""
    if arg is None or arg == "":
        return None
    names = model.names
    if arg.isdigit():
        return int(arg)
    for idx, name in names.items():
        if name == arg:
            return idx
    raise ValueError(f"未知类别名: {arg}，可用: {list(names.values())}")


def list_images(source, limit):
    """收集输入图片（单文件或目录）"""
    src = Path(source)
    if src.is_dir():
        imgs = [p for p in sorted(src.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        return imgs[:limit]
    return [src]


def find_per_class_images(source_dir, labels_dir, nc):
    """从 labels 目录里为每个类别挑第一张包含该类标注的图片，返回 [(class_id, img_path), ...]"""
    src = Path(source_dir)
    lab = Path(labels_dir)
    picked = {}
    for label_path in sorted(lab.glob("*.txt")):
        if len(picked) >= nc:
            break
        txt = label_path.read_text().strip()
        if txt == "":
            continue
        cls_ids = set()
        for line in txt.splitlines():
            parts = line.split()
            if parts:
                cls_ids.add(int(float(parts[0])))
        img_path = src / label_path.with_suffix(".jpg").name
        if not img_path.exists():
            continue
        for c in cls_ids:
            if c not in picked:
                picked[c] = img_path
        if len(picked) >= nc:
            break
    return [(picked[c], c) for c in sorted(picked)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--labels", default=None, help="YOLO 标注目录（--per-class 时用）")
    ap.add_argument("--out", default="experiments/gradcam")
    ap.add_argument("--class", dest="cls", default=None, help="类别名或索引（默认全局最高分）")
    ap.add_argument("--per-class", action="store_true", help="每类各生成一张热力图")
    ap.add_argument("--layer", type=int, default=-2, help="目标层索引")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    detect = model.model
    class_id = resolve_class_id(model, args.cls)
    print(f"目标层: 索引 {args.layer} -> {type(detect.model[args.layer]).__name__}")
    print(f"类别: {model.names}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.per_class:
        if not args.labels:
            raise SystemExit("--per-class 需要 --labels 指定标注目录")
        pairs = find_per_class_images(args.source, args.labels, len(model.names))
        if not pairs:
            raise SystemExit("未能从 labels 里挑到任何图片")
        print(f"每类挑图: {len(pairs)} 张")
    else:
        pairs = [(img, class_id) for img in list_images(args.source, args.limit)]

    for img_path, cid in pairs:
        try:
            x, img_rgb = preprocess(img_path)
            cam, score = grad_cam(model, x, args.layer, class_id=cid)
            label = model.names.get(cid, "all") if cid is not None else "all"
            out_img = overlay_heatmap(img_rgb, cam)
            save_path = out_dir / f"gradcam_{label}_{img_path.stem}.jpg"
            cv2.imwrite(str(save_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"[OK] {label:12s} {img_path.name}  (score={score:.3f}) -> {save_path.name}")
        except Exception as e:
            print(f"[ERR] {img_path.name}: {e}")

    print(f"\n完成，热力图保存在: {out_dir}")


if __name__ == "__main__":
    main()
