from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
from proxy_segearth_segmentor import ProxySegEarthSegmentation
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.patches as mpatches
import torch
from typing import Sequence

img_path = 'demo/image/austin_31.tif'
TARGET_PIXEL = (180, 336)          # (row, col) on 448×448 resized image
img = Image.open(img_path)
base_name = Path(img_path).stem  # 'kyoto_33'

name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
             'tree,forest', 'water,river', 'cropland', 'building,roof,house']

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()

pil = img.convert('RGB')
pil_resized = pil.resize((448, 448), Image.Resampling.BILINEAR)

img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((448, 448))
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')

model = ProxySegEarthSegmentation(
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
    ignore_residual=True,
    feature_up=True,
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
    cls_token_lambda=-0.3,
    name_path='./configs/my_name.txt',
    prob_thd=0.1,
    cls_variant="none",
    vfm_model="dino"
)

with torch.no_grad():
    #probs = model.predict_prob(img_tensor, data_samples=None)[0]     # [C,H,W]
    logits = model.predict_logit(img_tensor, data_samples=None)[0]     # [C,H,W]

# # pixel location visualization
# row, col = TARGET_PIXEL
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.imshow(pil_resized)
# ax.scatter([col], [row], c='red', s=40, marker='x')
# ax.set_title(f'Pixel ({row},{col})')
# ax.axis('off')
# fig.tight_layout()
# #vis_path = Path(IMG_PATH).with_name(f"{Path(IMG_PATH).stem}_pixel_{row}_{col}.png")
# vis_path = Path(f"visualize/logit/{Path(img_path).stem}_pixel_{row}_{col}.png")
# fig.savefig(vis_path, dpi=200)
# plt.close(fig)
# print(f"[INFO] saved visualization → {vis_path}")

# pixel_logits = logits[:, row, col]     # 확률 합 = 1
# print(f"\n[Pixel ({row},{col}) class logits]")
# for idx, p in enumerate(pixel_logits):
#     print(f"{idx:2d} – {name_list[idx]:25s}: {p.item()*100:6.2f}")

def plot_class_logit_along_line(
    logits: torch.Tensor | np.ndarray,
    class_idx: int,
    row_idx: int | None = None,
    col_idx: int | None = None,
    name_list: list[str] | None = None,
    title_suffix: str = "",
):
    """
    logits   : (C, H, W) 형태의 텐서 ─ softmax 전/후 모두 가능
    class_idx: 시각화할 클래스 인덱스
    row_idx  : 특정 행을 따라 그리려면 row_idx 지정 (0 ≤ row_idx < H)
    col_idx  : 특정 열을 따라 그리려면 col_idx 지정 (0 ≤ col_idx < W)
               row_idx 또는 col_idx 중 하나만 선택해야 함
    name_list: 인덱스 → 클래스 이름 매핑 리스트(선택)
    """

    # ─── 안전 장치 ────────────────────────────────────────────────────────
    assert (row_idx is None) ^ (col_idx is None), \
        "row_idx 또는 col_idx 둘 중 하나만 지정해야 합니다."
    if isinstance(logits, torch.Tensor):
        data = logits.cpu().detach().numpy()
    else:
        data = logits  # 이미 numpy 라면 그대로

    C, H, W = data.shape
    if class_idx >= C:
        raise ValueError(f"class_idx {class_idx} (C={C}) 범위를 벗어났습니다.")

    # ─── 라인 선택 & 값 추출 ────────────────────────────────────────────
    if row_idx is not None:
        if not (0 <= row_idx < H):
            raise ValueError(f"row_idx {row_idx} (H={H}) 범위를 벗어났습니다.")
        y_vals = data[class_idx, row_idx, :]  # 길이 W
        x_vals = np.arange(W)
        x_label = "Column index"
        title_core = f"Row {row_idx}"
    else:  # col_idx 모드
        if not (0 <= col_idx < W):
            raise ValueError(f"col_idx {col_idx} (W={W}) 범위를 벗어났습니다.")
        y_vals = data[class_idx, :, col_idx]  # 길이 H
        x_vals = np.arange(H)
        x_label = "Row index"
        title_core = f"Column {col_idx}"

    class_name = (
        f"{class_idx} – {name_list[class_idx]}" if name_list else f"Class {class_idx}"
    )

    # ─── 그래프 그리기 ──────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, marker="o", linewidth=1.2)
    plt.title(f"{class_name} logits along {title_core}{title_suffix}")
    plt.xlabel(x_label)
    plt.ylabel("Logit" if y_vals.max() > 1 else "Probability")
    plt.grid(True)
    plt.tight_layout()
    vis_path = Path(f"visualize/logit/{Path(img_path).stem}_{title_core}_{class_idx}.png")
    vis_path.parent.mkdir(parents=True, exist_ok=True)  # 폴더 자동 생성
    plt.savefig(vis_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_two_class_logits_along_line(
    logits: torch.Tensor | np.ndarray,
    class_indices: Sequence[int],        # 반드시 길이 2
    row_idx: int | None = None,
    col_idx: int | None = None,
    img_path: str = "image.png",
    name_list: list[str] | None = None,
):
    """
    logits        : (C, H, W) 텐서 (softmax 전/후 상관없음)
    class_indices : 길이 2의 시퀀스. 예) (3, 7)  또는 [5, 9]
    row_idx / col_idx : 둘 중 하나만 지정 (라인을 따라 추출)
    img_path      : 저장 파일명에 사용할 원본 이미지 경로
    name_list     : (선택) 클래스 인덱스 → 이름 매핑
    """
    # ─── 검증 ───────────────────────────────────────────────────────
    assert len(class_indices) == 2, "`class_indices`는 반드시 두 개여야 합니다."
    if (row_idx is None) == (col_idx is None):
        raise ValueError("row_idx 또는 col_idx 중 하나만 지정해야 합니다.")

    # numpy 로 변환
    data = logits.cpu().detach().numpy() if isinstance(logits, torch.Tensor) else logits
    C, H, W = data.shape

    for c in class_indices:
        if not (0 <= c < C):
            raise ValueError(f"class_idx {c} 는 범위(C={C})를 벗어났습니다.")

    # ─── 라인 추출 ──────────────────────────────────────────────────
    if row_idx is not None:
        if not (0 <= row_idx < H):
            raise ValueError(f"row_idx {row_idx} 는 H={H} 범위를 벗어났습니다.")
        y_vals = [data[c, row_idx, :] for c in class_indices]   # 길이 W
        x_vals = np.arange(W)
        x_label = "Column index"
        title_core = f"Row{row_idx}"
    else:  # col_idx 지정
        if not (0 <= col_idx < W):
            raise ValueError(f"col_idx {col_idx} 는 W={W} 범위를 벗어났습니다.")
        y_vals = [data[c, :, col_idx] for c in class_indices]   # 길이 H
        x_vals = np.arange(H)
        x_label = "Row index"
        title_core = f"Col{col_idx}"

    # ─── 그래프 ────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))

    for y, c in zip(y_vals, class_indices):
        label = f"{c} – {name_list[c]}" if name_list else f"Class {c}"
        plt.plot(x_vals, y, marker="o", markersize=1.5, linewidth=1.2, label=label)

    plt.title(f"Logits along {title_core}")
    plt.xlabel(x_label)
    plt.ylabel("Logit" if max(map(np.max, y_vals)) > 1 else "Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ─── 저장 ───────────────────────────────────────────────────────
    vis_name = (
        f"{Path(img_path).stem}_{title_core}_C{class_indices[0]}_C{class_indices[1]}.png"
    )
    vis_path = Path("visualize/logit") / vis_name
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(vis_path, dpi=300, bbox_inches="tight")
    plt.close()

# plot_class_logit_along_line(logits, class_idx=1, col_idx=300, name_list=name_list)
# plot_class_logit_along_line(logits, class_idx=6, col_idx=300, name_list=name_list)
plot_two_class_logits_along_line(
    logits,
    class_indices=(3, 8),
    row_idx = 180,
    img_path=img_path,
    name_list=name_list,
)