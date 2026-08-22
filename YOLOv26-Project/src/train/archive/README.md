# 归档训练脚本（历史实验，勿再使用）

本目录存放已被 `train_final.py` 取代的历史训练脚本，仅作结题报告/论文「实验过程」证据保留。

> **当前唯一在用的训练脚本：`src/train/train_final.py`**（yolov8s + data_final.yaml，产出 `experiments/yolov26_baseline_v2`，最终 mAP50 = 0.424）。

| 脚本 | 说明 | 结论 |
|------|------|------|
| `train_baseline.py` | 最初基线（yolov8n / 8 类 / data.yaml） | ❌ 被 train_final 取代 |
| `train_960.py` | 960 分辨率微调 | ➖ 打平（0.421），放弃 |
| `train_p2.py` | P2/4 小目标检测头 | ❌ 0.357，失败，放弃 |
| `train_improved.py` | 消融实验（C2fLite 等改进模块） | 验证后未采用 |
| `train_crack_v3/v4` | 早期裂缝数据扩充迭代（8 类） | ❌ 被取代 |
| `train_v4/v6/v7` | 早期 v 系列迭代（6~8 类） | ❌ 被取代 |
| `train_yolov26/s` | 早期「YOLOv26」命名迭代 | ❌ 被取代 |

## 为什么归档而不是删除
这些脚本连同各自 `experiments/*/args.yaml` 是「数据清理 8→7 类才涨点、P2 失败、960 打平」等结论的**过程证据**，结题报告/论文需要引用。无 git 历史时删除不可恢复，故移到此目录保留。
