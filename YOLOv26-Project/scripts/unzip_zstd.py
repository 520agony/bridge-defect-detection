"""用 pyzipper(非AES) + zstd 解压 method 99 加密 zip"""
import os, sys

# Monkey-patch zipfile._get_decompressor for method 99
import zipfile as _base_zip

class _NoopDecompressor:
    eof = False
    unused_data = b''
    def decompress(self, data, max_length=0):
        return data

_orig_gd = _base_zip._get_decompressor
_base_zip._get_decompressor = lambda ct: _NoopDecompressor() if ct == 99 else _orig_gd(ct)

import pyzipper
import zstandard as zstd

ZIP_PATH = "datasets/images/混凝土表面裂缝目标检测数据集/混凝土表面裂缝目标检测 数据集/混凝土表面裂缝目标检测数据集-解压后使用.zip"
DST_DIR  = "datasets/images/混凝土表面裂缝目标检测数据集/_extracted"
PASSWORD = "fJR5Rv3zyu"

os.makedirs(DST_DIR, exist_ok=True)
dctx = zstd.ZstdDecompressor()

# 先试 ZipFile (ZipCrypto)，再试 AESZipFile
enc_types = [
    ("ZipCrypto", lambda: pyzipper.ZipFile(ZIP_PATH, "r")),
    ("AES",       lambda: pyzipper.AESZipFile(ZIP_PATH, "r")),
]

for enc_name, open_fn in enc_types:
    print(f"Trying {enc_name}...")
    try:
        zf = open_fn()
        zf.setpassword(PASSWORD.encode())
        # 测试读第一个非目录文件
        for info in zf.infolist():
            if not info.is_dir():
                raw = zf.read(info.filename)
                print(f"  OK! First file: {info.filename}, size={len(raw)} compress_type={info.compress_type}")
                break
        zf.close()
        print(f"  {enc_name} works!")
        break
    except Exception as e:
        print(f"  {enc_name}: {e}")
else:
    print("Both failed! Trying alternative approach...")

    # Alternative: use subprocess with unzip to decrypt, pipe through zstd
    pass
