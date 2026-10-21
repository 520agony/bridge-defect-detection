"""
make_server_pack.py - 打云端提交包（本机跑）
============================================
产出（项目根目录下 server_pack/）：
  code.zip            代码+配置+权重（小包，先传）
  final_5class.zip    数据集（大包，后传）
上传后云上：解压 -> bash scripts/server_setup.sh

跑法：.venv/Scripts/python.exe scripts/make_server_pack.py
"""
import os
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "server_pack"
OUT.mkdir(exist_ok=True)

CODE_DIRS = ["scripts", "src", "configs", "models"]
CODE_FILES = ["requirements-server.txt"]
WEIGHTS = ["weights/yolo26s-seg.pt", "weights/yolo26m-seg.pt"]
SKIP_SUFFIX = (".pyc", ".cache", ".log")


def add_tree(z, rel_dir):
    for root, _, files in os.walk(BASE / rel_dir):
        for f in files:
            if f.endswith(SKIP_SUFFIX) or "__pycache__" in root:
                continue
            p = Path(root) / f
            z.write(p, p.relative_to(BASE))


def make_zip(name, func, compress=True):
    # JPEG 数据压不动还耗时，大包用 STORED 只打包不压缩
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    zp = OUT / name
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", mode, compresslevel=6) as z:
        func(z)
    print(f"{name}: {zp.stat().st_size / 1024**2:.1f} MB", flush=True)
    return zp


def code_fn(z):
    for d in CODE_DIRS:
        add_tree(z, d)
    for f in CODE_FILES + WEIGHTS:
        z.write(BASE / f, f)


def data_fn(z):
    add_tree(z, "datasets/final_5class")


if __name__ == "__main__":
    import sys
    # 用法：make_server_pack.py [code|data|all]，默认 all
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("code", "all"):
        make_zip("code.zip", code_fn, compress=True)
    if which in ("data", "all"):
        make_zip("final_5class.zip", data_fn, compress=False)
    print("\n上传顺序：1) code.zip  2) final_5class.zip")
    print("云上：unzip code.zip && unzip final_5class.zip && bash scripts/server_setup.sh")
