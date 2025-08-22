from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
from proxy_segearth_segmentor import ProxySegEarthSegmentation
import os
import matplotlib.patches as mpatches

# === 1. 경로 설정 ===
IMG_DIR  = Path("demo/image")
OUT_DIR  = Path("visualize/ProxyCLIP")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 2. CLIP prompt용 이름 리스트 작성 ===
name_list = [
    'background',
    'bareland,barren',
    'grass',
    'pavement',
    'road',
    'tree,forest',
    'water,river',
    'cropland',
    'building,roof,house'
]
os.makedirs("configs", exist_ok=True)
with open("configs/my_name.txt", "w") as f:
    f.write("\n".join(name_list))

# === 3. 모델 공통 파라미터 정의 ===
common_kwargs = dict(
    clip_type='CLIP',
    vit_type='ViT-B/16',
    model_type='SegEarth',
    ignore_residual=True,
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'
    ),
    cls_token_lambda=-0.3,
    name_path='./configs/my_name.txt',
    prob_thd=0.1,
    cls_variant="none",
    vfm_model="dino"
)

# === 4. 전처리 정의 (448×448) ===
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
    ),
    transforms.Resize((448, 448))
])

# === 5. OpenEarthMap 9-클래스 컬러맵 정의 ===
class_info = {
    0: ("Background",        "#000000"),
    1: ("Bareland",          "#800000"),
    2: ("Rangeland",         "#00FF24"),
    3: ("Developed space",   "#949494"),
    4: ("Road",              "#FFFFFF"),
    5: ("Tree",              "#226126"),
    6: ("Water",             "#0045FF"),
    7: ("Agriculture land",  "#4BB549"),
    8: ("Building",          "#DE1F07"),
}
colors = [mcolors.hex2color(class_info[i][1]) for i in range(9)]
cmap_9 = mcolors.ListedColormap(colors)
norm_9 = mcolors.BoundaryNorm(np.arange(10) - 0.5, cmap_9.N)

# === 6. 시각화 함수 ===
def save_mask_with_legend(mask, filepath, title):
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(mask, cmap=cmap_9, vmin=0, vmax=8)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    
    boundaries = np.arange(10) - 0.5  # [-0.5,0.5,...,8.5]
    cbar = plt.colorbar(
        im,
        ax=ax,
        ticks=np.arange(9),
        boundaries=boundaries,
        fraction=0.046,
        pad=0.04
    )
    cbar.ax.set_yticklabels([class_info[i][0] for i in range(9)])
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)

def save_overlay(img, mask, filepath, alpha=0.25):
    fig, ax = plt.subplots(figsize=(8, 8))
    # 원본 크기→(448,448) 리사이즈
    resized = img.resize(mask.shape[::-1], resample=Image.Resampling.BILINEAR)
    ax.imshow(resized)
    ax.imshow(mask, cmap=cmap_9, norm=norm_9, alpha=alpha)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close(fig)

def save_overlay_with_legend(img, seg_mask, filename, name_list):
    # 1) 준비
    fig, ax = plt.subplots(figsize=(8, 8))
    # 2) 원본 리사이즈
    resized_img = img.resize((448, 448), resample=Image.Resampling.BILINEAR)
    ax.imshow(resized_img)
    # 3) 분할 colormap
    cmap = plt.get_cmap('tab20', len(name_list))
    ax.imshow(seg_mask,
              cmap=cmap,
              alpha=0.5,
              vmin=0, vmax=len(name_list)-1)

    # 5) legend 추가
    patches = []
    for idx, full_name in enumerate(name_list):
        label = full_name.split(',')[0]
        color = cmap(idx)
        patches.append(mpatches.Patch(color=color, label=label))
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1.0),
        loc='upper left',
        fontsize=8,
        frameon=False
    )

    # 6) 마무리 저장
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

# === 7. 메인 루프: 모든 이미지 → feature_up True/False → 저장 ===
for img_path in sorted(IMG_DIR.glob("*.tif")):
    base = img_path.stem
    print(f">>> Processing {base} …")
    # 7.1 원본 이미지 로드 & 텐서화
    img = Image.open(img_path)
    img_tensor = preprocess(img).unsqueeze(0).to("cuda")

    # for feat_up in [True, False]:
    for feat_up in [True]:
        # 7.2 모델 초기화 (feature_up 설정)
        model = ProxySegEarthSegmentation(
            feature_up=feat_up,
            **common_kwargs
        )
        # 7.3 예측 수행
        seg_pred = model.predict(img_tensor, data_samples=None)
        mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

        # 7.4 파일명 suffix
        suffix = "_featup" if feat_up else "_nofeatup"
        title  = f"OpenEarthMap - {base}{suffix}"
        mask_file    = OUT_DIR / f"{base}{suffix}_pred.png"
        overlay_file = OUT_DIR / f"{base}{suffix}_pred_overlay.png"

        # 7.5 저장
        save_mask_with_legend(mask, str(mask_file), title)
        #save_overlay(img, mask, str(overlay_file))
        save_overlay_with_legend(
            img,
            mask,
            filename=f"visualize/prediction/{base}{suffix}_pred_overlay.png",
            name_list=name_list
        )

print("✅ All done!")
