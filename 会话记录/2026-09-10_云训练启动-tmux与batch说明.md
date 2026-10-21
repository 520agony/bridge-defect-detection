# 会话记录 — 2026-09-10 夜：AutoDL 云训练正式启动（tmux + batch）

## 1. 项目背景（给新会话看）
- 智巡飞眼大创：5类桥梁病害 YOLO 分割（crack 裂缝 / hole 孔洞 / spalling 剥落 / rebar_exposed 露筋 / efflorescence 泛碱）+ 前端平台 + 云端训练。
- 建模路线：先稳 2D seg，点云后融合；LiDAR 不买（借/租）；图像机 Mini 4 Pro 或 Air 3S。
- 数据：`YOLOv26-Project/datasets/final_5class/`，train 32263 张。val/test 与 train 同源泄漏约 12.9%/14.7%，云上 best 会偏乐观，结题前要洗独立测试集。
- 本机 RTX 5060 Laptop 8G；云端 AutoDL 重庆A区 RTX 4090D 1卡 ¥1.88/时，余额约 ¥200。
- 云配置：镜像 PyTorch 2.8.0 + Python 3.12 Ubuntu22.04；工作目录 `/root/autodl-tmp`；ultralytics 钉死 8.4.115。

## 2. 本次会话做了什么
1. 确认正式训练已启动：`python scripts/cloud_train.py --mode s`（200ep, batch32, imgsz1024, copy_paste=0.3），截图 `Epoch 1/200`、GPU_mem 14.4G、6.4it/s。
2. 回答用户三问：
   - 跑起来了（Epoch 1/200 + 刷 loss 即正常；corrupt/duplicate 警告可忽略）。
   - 关网页/本机：训练在云上，但必须在 tmux 里才能关。直接在 Jupyter 终端跑则不能关。
   - 时长/费用：冒烟 1 轮约 14 分钟，200 轮约 35–45 小时（patience=30 可能提前停）；¥1.88/时约 ¥85，200 元够跑完 s 基线，m/distill 等这轮完再开。
3. 查 tmux：`tmux ls` 报 `command not found`，说明镜像没装 tmux，且之前不在 tmux 里。
4. 装 tmux：`apt-get update && apt-get install -y tmux`。
5. 进 tmux 重开训练（因之前才第 1 轮，重来无损失）：
   ```bash
   cd /root/autodl-tmp
   tmux new -s train
   python scripts/cloud_train.py --mode s
   ```
   底栏出现绿色 `[train]`，`[train] 0:python*` + `Epoch 1/200` 刷 loss，确认在 tmux 里跑着。
6. 教 detach：`Ctrl+B` 松开再按 `D`，之后可关网页；回来 `tmux attach -t train`。
7. 回答 batch：s=32，m=16，distill=32（见 `scripts/cloud_train.py:29-41`）。24G 卡可加到 40（稳）/48（上限，val 可能 OOM），脚本支持 `--batch` 覆盖，但要 `Ctrl+C` 重训。

## 3. 当前状态 / 待确认（重要）
- 训练正在 `train` 会话里跑 Epoch 1/200，但用户按 `Ctrl+B,D` 后的截图底栏仍显示 `[train] 0:python*`，**疑似没 detach 成功**（正常应显示 `[detached]` 并回到普通命令行）。下次开会话先让用户在新终端跑 `tmux ls` / `tmux attach -t train` 确认。
- batch 仍是 32，未加大。用户问 24G 是否调大，已给方案，未决定是否重训。
- AutoDL 控制台别点关机/释放；关机数据保留，释放全没。

## 4. 命令速查（云上）
```bash
cd /root/autodl-tmp
tmux ls                          # 查会话
tmux attach -t train             # 回看训练
# detach：Ctrl+B 松开再按 D
python scripts/cloud_train.py --mode s                 # s 基线（batch32）
python scripts/cloud_train.py --mode s --batch 48      # 加 batch（需先 Ctrl+C 停掉）
python scripts/cloud_train.py --mode s --resume        # 断点续训
```

## 5. 相关文件（本机）
- `YOLOv26-Project/scripts/cloud_train.py`：云训练 s/m/distill，batch 定义 29–41 行
- `YOLOv26-Project/configs/data_cloud_5class.yaml`：5 类数据配置
- `YOLOv26-Project/scripts/server_setup.sh`：云上一键环境+冒烟
- `YOLOv26-Project/server_pack/`：code.zip / final_5class.zip(8.5G) / smoke_data.zip
- `YOLOv26-Project/src/fusion/late_fusion.py` + `selftest.py`：后融合（11 项已过）
- 云上输出：`/root/autodl-tmp/experiments/cloud_s/weights/best.pt`（训完下载看每类 mAP50）
