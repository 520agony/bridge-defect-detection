#!/bin/bash
# 云服务器一键初始化（AutoDL PyTorch 镜像，项目根目录执行）
#   bash scripts/server_setup.sh
set -e
cd "$(dirname "$0")/.."

echo "[1/4] 安装依赖..."
pip install -r requirements-server.txt

echo "[2/4] 检查数据与权重..."
test -d datasets/final_5class/images/train || { echo "缺 datasets/final_5class，请先解压 final_5class.zip"; exit 1; }
test -f weights/yolo26s-seg.pt || { echo "缺 weights/yolo26s-seg.pt"; exit 1; }
test -f weights/yolo26m-seg.pt || { echo "缺 weights/yolo26m-seg.pt"; exit 1; }
python -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__)"

echo "[3/4] 冒烟测试（1 epoch）..."
python scripts/cloud_train.py --mode s --smoke

echo "[4/4] OK。正式训练三步走："
echo "  python scripts/cloud_train.py --mode s        # ① s 基线"
echo "  python scripts/cloud_train.py --mode m        # ② m 教师"
echo "  python scripts/cloud_train.py --mode distill  # ③ 蒸馏（需②的 best.pt）"
echo "长任务请用 tmux 跑：tmux new -s train"
