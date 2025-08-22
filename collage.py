# import matplotlib.pyplot as plt
# from PIL import Image

# # 이미지 파일 경로
# paths = {
#     "blur": "demo/image/kyoto_33.tif",
#     "overlay": "visualize/prediction/kyoto_33_pred_overlay.png",
#     "label": "visualize/label/kyoto_33_label.png",
#     "mask": "visualize/prediction/kyoto_33_pred.png",
# }

# imgs = [Image.open(paths[k]).convert("RGB") for k in ("blur","overlay","label","mask")]

# # 2) 네 이미지 중 가장 작은 세로 높이
# min_h = min(im.height for im in imgs)

# # 3) 세로 높이만 맞추고 가로 비율 유지하며 리사이즈
# resized = []
# for im in imgs:
#     w, h = im.size
#     new_w = int(w * min_h / h)
#     resized.append(im.resize((new_w, min_h), resample=Image.Resampling.LANCZOS))

# # 4) 캔버스 크기 계산 (위쪽 행: blur + label, 아래쪽 행: overlay + mask)
# w_top    = resized[0].width + resized[2].width   # blur + label
# w_bottom = resized[1].width + resized[3].width   # overlay + mask
# canvas_w = max(w_top, w_bottom)
# canvas_h = 2 * min_h

# # 5) 빈 캔버스 생성 (흰 배경)
# collage = Image.new("RGB", (canvas_w, canvas_h), (255,255,255))

# # 6) 붙여넣기 (label ↔ overlay 위치만 스왑)
# # top-left: blur
# collage.paste(resized[0], (0, 0))
# # top-right: label
# collage.paste(resized[2], (resized[0].width, 0))

# # bottom-left: overlay
# collage.paste(resized[1], (0, min_h))
# # bottom-right: mask
# collage.paste(resized[3], (resized[1].width, min_h))

# # 7) 결과 저장 및 보여주기
# collage.save("visualize/variants/collage/kyoto_33_original.png")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make 2×2 collages for every *type* found in visualize/variants/.
Layout (top-left → clockwise):
    blur | label
    overlay | mask
Output:
    visualize/variants/collage/kyoto_33_{type}_collage.png
"""

import os
import re
import glob
from PIL import Image

# ----------------------------------------------------------------------
# 0. 고정 경로 설정
LABEL_PATH = "visualize/label/kyoto_33_label.png"             # (고정)
VAR_DIR    = "visualize/variants"                             # blur / overlay / mask 들
OUT_DIR    = os.path.join(VAR_DIR, "collage")                 # 저장 폴더
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """세로 높이만 target_h 로 맞추고 가로는 비율 유지."""
    w, h = img.size
    new_w = int(w * target_h / h)
    return img.resize((new_w, target_h), resample=Image.Resampling.LANCZOS)

def make_collage(blur_path: str, overlay_path: str, mask_path: str,
                 label_path: str, save_path: str) -> None:
    """네 장(blur, overlay, label, mask)을 2×2 콜라주로 저장."""
    imgs = [Image.open(p).convert("RGB")
            for p in (blur_path, overlay_path, label_path, mask_path)]

    # 가장 작은 세로 픽셀 수에 맞춰 리사이즈
    min_h = min(im.height for im in imgs)
    resized = [resize_to_height(im, min_h) for im in imgs]

    # 캔버스 크기 (위: blur+label, 아래: overlay+mask)
    w_top    = resized[0].width + resized[2].width
    w_bottom = resized[1].width + resized[3].width
    canvas_w = max(w_top, w_bottom)
    canvas_h = 2 * min_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    # ↖ blur
    canvas.paste(resized[0], (0, 0))
    # ↗ label
    canvas.paste(resized[2], (resized[0].width, 0))
    # ↙ overlay
    canvas.paste(resized[1], (0, min_h))
    # ↘ mask
    canvas.paste(resized[3], (resized[1].width, min_h))

    canvas.save(save_path)
    print(f"✔ saved: {save_path}")

# ----------------------------------------------------------------------
# 1. variants 폴더에서 blur 파일(=기준 파일) 찾기
blur_files = glob.glob(os.path.join(VAR_DIR, "kyoto_33_*.png"))

# 패턴: kyoto_33_{type}.png (overlay/mask/… 제외)
blur_re = re.compile(r"kyoto_33_(.+)\.png$")

for blur_path in blur_files:
    fname = os.path.basename(blur_path)
    # overlay/mask 는 continue
    if fname.endswith("_overlay.png") or fname.endswith("_mask.png"):
        continue

    m = blur_re.match(fname)
    if not m:
        continue
    type_str = m.group(1)                # 예: blur_s4-0, sharp, original … 등

    # 필요한 파일 경로 생성
    overlay_path = os.path.join(VAR_DIR, f"kyoto_33_{type_str}_overlay.png")
    mask_path    = os.path.join(VAR_DIR, f"kyoto_33_{type_str}_mask.png")

    # 존재 여부 확인
    if not (os.path.exists(overlay_path) and os.path.exists(mask_path)):
        print(f"⚠  skip {type_str}: overlay/mask 파일이 없습니다")
        continue
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"label 이미지가 없습니다: {LABEL_PATH}")

    # 저장 경로
    save_path = os.path.join(OUT_DIR, f"kyoto_33_{type_str}_collage.png")

    # 콜라주 생성
    make_collage(blur_path, overlay_path, mask_path, LABEL_PATH, save_path)
