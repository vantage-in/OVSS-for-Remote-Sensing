from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
# from proxy_segearth_segmentor import ProxySegEarthSegmentation
from proxy_segearth_segmentor_cat import ProxySegEarthSegmentationCat   
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.patches as mpatches
import torch

img_path = 'data/iSAID/img_dir/val/P0003_0_896_0_896.png'
img = Image.open(img_path)
base_name = Path(img_path).stem  # 'kyoto_33'

name_list = ['background', 'ship', 'store tank', 'baseball diamond', 'tennis court',
    'basketball court', 'ground track field', 'bridge', 'large vehicle', 'small vehicle',
    'helicopter', 'swimming pool', 'roundabout', 'soccer ball field', 'plane', 'harbor']

with open('./configs/my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()


img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((448, 448))
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')

model = ProxySegEarthSegmentationCat(
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
    prob_thd=0.4,
    cls_variant="none",
    vfm_model="dino"
)

seg_pred = model.predict(img_tensor, data_samples=None)
seg_mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

class_info = {
    0: ("background", "#000000"),          # 검정
    1: ("ship", "#000080"),                # 네이비
    2: ("store tank", "#008080"),          # 청록 (Teal)
    3: ("baseball diamond", "#FFFF00"),    # 노랑
    4: ("tennis court", "#ADFF2F"),        # 연두 (GreenYellow)
    5: ("basketball court", "#FFA500"),    # 주황
    6: ("ground track field", "#FF0000"),  # 빨강
    7: ("bridge", "#A9A9A9"),              # 회색 (DarkGray)
    8: ("large vehicle", "#8A2BE2"),       # 보라 (BlueViolet)
    9: ("small vehicle", "#D2691E"),       # 갈색 (Chocolate)
    10: ("helicopter", "#DC143C"),         # 진홍 (Crimson)
    11: ("swimming pool", "#87CEEB"),     # 하늘색
    12: ("roundabout", "#FFC0CB"),         # 핑크
    13: ("soccer ball field", "#008000"), # 초록
    14: ("plane", "#0000FF"),             # 파랑
    15: ("harbor", "#800000")              # 밤색 (Maroon)
}

colors = [mcolors.hex2color(class_info[i][1]) for i in range(len(class_info))]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(len(class_info) + 1) - 0.5, cmap.N)

# === 7. Functions to save visualizations ===
def save_mask_with_legend(mask, filename, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mask, cmap=cmap, norm=norm)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(16),
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([class_info[i][0] for i in range(16)])
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
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
    
# === 8. Save the outputs ===
if model.feature_up:
    pred_path = Path("visualize/cat") / f"0820_{base_name}_pred.png"
    overlay_path = Path("visualize/cat") / f"0820_{base_name}_pred_overlay.png"
else:
    pred_path = Path("visualize/prediction") / f"08132_{base_name}_no_pred.png"
    overlay_path = Path("visualize/prediction") / f"08132_{base_name}_no_pred_overlay.png"

save_mask_with_legend(
    seg_mask,
    filename=str(pred_path),
    title=f"OpenEarthMap - {base_name}"
)
# save_overlay_with_legend(
#     img,
#     seg_mask,
#     filename=str(overlay_path),
#     name_list=name_list
# )


# === 사용자 입력 ===
label_path = "data/iSAID/ann_dir/val/P0003_0_896_0_896_instance_color_RGB.png"  # 단일 채널 label mask 경로

# === 시각화를 위한 color map 생성 ===
colors = [mcolors.hex2color(class_info[i][1]) for i in range(16)]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(17) - 0.5, cmap.N)

# === label mask 불러오기 ===
label_img = Image.open(label_path)
label_array = np.array(label_img)

# === 파일 이름 추출 ===
filename = Path(label_path).stem

# === 시각화 ===
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(label_array, cmap=cmap, norm=norm)
ax.axis('off')
ax.set_title(f"OpenEarthMap - {filename}", fontsize=14)

# colorbar 추가 (이미지 세로 높이 맞춤)
cbar = plt.colorbar(im, ax=ax, ticks=np.arange(16), fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels([class_info[i][0] for i in range(16)])
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
pth = f'./visualize/label/{filename}_label.png'
plt.savefig(pth, bbox_inches='tight')
