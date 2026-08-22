"""
YOLOv26 改进模块注册 + 构建脚本
================================
将 C2fLite / HDDP / CoordinateAttention 注册到 ultralytics.nn.tasks，
使它们能在 model YAML 中被 parse_model 解析。

使用方式（关键）：
    import register_improvements   # 先 import，完成模块注册
    from ultralytics import YOLO
    model = YOLO("models/yolov26_improved.yaml")

原理：
    ultralytics.nn.tasks.parse_model 里通过 `globals()[m]` 解析模块名，
    因此只需把自定义类注入到 tasks 模块的命名空间即可。

注意（通道传递逻辑）：
    parse_model 的 `else` 分支对未知模块只做 `c2 = ch[f]`，
    即「输出通道 = 上一层输出通道」。因此：
    - C2fLite(in, out, n)  必须手动处理 in/out —— 已按 C2f 风格实现
    - HDDP(in, out)        必须手动处理 in/out
    - CoordinateAttention(ch)  通道不变，输入输出一致
    在 YAML 里 args 需要显式写明 [in, out]。
"""

import torch.nn as nn

import ultralytics.nn.tasks as tasks

from src.models.yolov26_improvements import (
    C2fLite,
    HDDP,
    CoordinateAttention,
    LeakyReLU6,
    cpu_friendly_nms,
    soft_nms,
)


# ============================================================
# 注册自定义模块到 ultralytics.nn.tasks 命名空间
# ============================================================
def register_all():
    """把自定义模块注入 tasks 模块的 globals，供 parse_model 使用。"""
    tasks.C2fLite = C2fLite
    tasks.HDDP = HDDP
    tasks.CoordinateAttention = CoordinateAttention
    tasks.LeakyReLU6 = LeakyReLU6
    print("[OK] 已注册自定义模块: C2fLite, HDDP, CoordinateAttention, LeakyReLU6")


# 在 import 本文件时自动注册
register_all()


if __name__ == "__main__":
    from ultralytics import YOLO
    import sys

    # 快速自检：加载改进版 YAML 并打印模型结构
    yaml_path = "models/yolov26_improved.yaml"
    print(f"加载模型配置: {yaml_path}")
    model = YOLO(yaml_path)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"\n模型参数总量: {n_params / 1e6:.2f} M")
    print("\n改进模块已正确接入 Ultralytics 解析流程 ✓")
