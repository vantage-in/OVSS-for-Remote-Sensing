import os
import shutil

# 원본 이미지, 레이블, 타겟 폴더 경로
IMG_DIR   = "demo/image"
LABEL_DIR = "data/OpenEarthMap/ann_dir/val"
DST_DIR   = "demo/label"

# demo/label 폴더가 없으면 생성
os.makedirs(DST_DIR, exist_ok=True)

for fname in os.listdir(IMG_DIR):
    if not fname.lower().endswith(".tif"):
        continue

    src_lbl = os.path.join(LABEL_DIR, fname)
    dst_lbl = os.path.join(DST_DIR, fname)

    if os.path.isfile(src_lbl):
        shutil.copy(src_lbl, dst_lbl)
        print(f"✔ copied: {fname}")
    else:
        print(f"✖ missing: {fname}")

print("Done.")
