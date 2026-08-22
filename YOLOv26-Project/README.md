# YOLOv26 桥梁病害检测项目

## 项目进度

| 阶段 | 状态 | 时间 |
|------|------|------|
| 环境搭建 | ✅ 完成 | 2026.08.06 |
| 基线训练 | ✅ 完成 | 2026.08.06 |
| 数据采集 | 🔲 待进行 | - |
| 模型改进 | 🔲 待进行 | - |
| Web 平台 | 🔲 待进行 | - |

## 环境

- Python 3.12.9 | PyTorch 2.7.1+cu128 | CUDA 12.8
- GPU: RTX 5060 Laptop (8GB)
- 虚拟环境: `.venv/Scripts/activate`

## 快速开始

```bash
# 激活环境
.venv\Scripts\activate

# 准备数据（合成数据验证流程）
python src/data/prepare_dataset.py --mode 1

# 训练基线模型
python src/train/train_baseline.py

# 仅验证
python src/train/train_baseline.py --val

# 测试改进模块
python src/models/yolov26_improvements.py
```

## 项目结构

```
YOLOv26-Project/
├── configs/data.yaml              # 数据集配置
├── datasets/                      # 数据集
│   ├── images/{train,val,test}/   # 图片
│   ├── labels/{train,val,test}/   # YOLO标注
│   └── lidar/{train,val,test}/    # LiDAR点云（待采集）
├── src/
│   ├── data/prepare_dataset.py    # 数据准备工具
│   ├── train/train_baseline.py    # 基线训练脚本
│   └── models/yolov26_improvements.py  # 改进模块
├── experiments/                   # 训练输出
├── models/                        # 模型配置 YAML
├── weights/                       # 预训练权重
├── web/{backend,frontend}/        # Web 平台（待开发）
└── docs/                          # 文档
```

## 基线结果

| 指标 | 数值 |
|------|------|
| mAP@0.5 | 0.861 |
| mAP@0.5:0.95 | 0.429 |
| Precision | 0.729 |
| Recall | 0.833 |
| F1 | 0.777 |
| 推理速度 | 4.6ms/img (217 FPS) |
| 模型大小 | 6.0 MB |
