"""
合并全部数据 → 按 7:1.5:1.5 重新划分 train/val/test，按类别比例分层。
"""
import os, shutil, random
from collections import Counter, defaultdict

random.seed(42)

BASE = "datasets"
OLD_IMG = f"{BASE}/images"
OLD_LBL = f"{BASE}/labels"
NEW_IMG = f"{BASE}/images_new"
NEW_LBL = f"{BASE}/labels_new"

# ============================================================
# 1. 收集所有图片+标注，记录每张图的类别
# ============================================================
records = []  # [(img_file, lbl_file, source_split)]

for split in ["train", "val", "test"]:
    img_dir = f"{OLD_IMG}/{split}"
    lbl_dir = f"{OLD_LBL}/{split}"
    for f in os.listdir(lbl_dir):
        if not f.endswith(".txt"):
            continue
        base = os.path.splitext(f)[0]
        # 找对应图片
        img_file = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            c = base + ext
            if os.path.exists(os.path.join(img_dir, c)):
                img_file = c
                break
        if img_file is None:
            continue
        # 判断是否为新增的混凝土裂缝数据
        is_new_crack = base.startswith("cc_") or base.startswith("crack500_")
        records.append({
            "img": (os.path.join(img_dir, img_file), os.path.join(NEW_IMG, split, img_file)),
            "lbl": (os.path.join(lbl_dir, f), os.path.join(NEW_LBL, split, f)),
            "source": split,
            "is_new_crack": is_new_crack,
            "base": base,
        })

print(f"总计: {len(records)} 条数据")

# 统计原分配的 new crack 分布
new_crack_splits = Counter()
for r in records:
    if r["is_new_crack"]:
        new_crack_splits[r["source"]] += 1
print(f"新裂缝数据原分布: {dict(new_crack_splits)}")

# ============================================================
# 2. 分成两组：原数据 / 新裂缝数据，各自 7:1.5:1.5 分层
# ============================================================
def split_group(items, train_r=0.7, val_r=0.15, test_r=0.15):
    """随机打散后按比例分"""
    items = list(items)
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    # 调整使总数为 n
    n_test = n - n_train - n_val
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }

orig = [r for r in records if not r["is_new_crack"]]
new_crack = [r for r in records if r["is_new_crack"]]

print(f"原数据: {len(orig)} 条")
print(f"新裂缝数据: {len(new_crack)} 条")

# 各自分层分
orig_split = split_group(orig)
new_split = split_group(new_crack)

# 合并
allocation = {}
for split in ["train", "val", "test"]:
    allocation[split] = orig_split[split] + new_split[split]
    print(f"{split}: {len(allocation[split])} 条 (原{len(orig_split[split])} + 新{len(new_split[split])})")

# ============================================================
# 3. 写入新目录
# ============================================================
for split in ["train", "val", "test"]:
    os.makedirs(f"{NEW_IMG}/{split}", exist_ok=True)
    os.makedirs(f"{NEW_LBL}/{split}", exist_ok=True)

for split in ["train", "val", "test"]:
    for r in allocation[split]:
        src_img, dst_img = r["img"]
        src_lbl, dst_lbl = r["lbl"]
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

print("\n写入完成！")

# ============================================================
# 4. 验证新划分
# ============================================================
names = {0: 'corrosion', 1: 'crack', 2: 'degraded_concrete', 3: 'concrete_hole',
         4: 'moisture', 5: 'pavement_degrade', 6: 'shrinkage_crack', 7: 'sub_shrinkage_crack'}

for split in ["train", "val", "test"]:
    lbl_dir = f"{NEW_LBL}/{split}"
    cls_cnt = Counter()
    for f in os.listdir(lbl_dir):
        if not f.endswith(".txt"): continue
        with open(os.path.join(lbl_dir, f)) as fh:
            for line in fh:
                parts = line.strip().split()
                if parts:
                    cls_cnt[int(float(parts[0]))] += 1
    n_files = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])
    n_imgs = len([f for f in os.listdir(f"{NEW_IMG}/{split}")
                  if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
    print(f"\n{split} ({n_imgs} imgs, {n_files} lbls):")
    for k in sorted(cls_cnt.keys()):
        print(f"  {names[k]:>20s}: {cls_cnt[k]:>6d}")
    print(f"  {'TOTAL':>20s}: {sum(cls_cnt.values()):>6d}")
