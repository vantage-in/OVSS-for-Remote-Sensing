from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
# from segearth_segmentor import SegEarthSegmentation
from proxy_segearth_segmentor import ProxySegEarthSegmentation
# from proxy_segearth_segmentor_cat import ProxySegEarthSegmentationCat   
# from proxy_segearth_segmentor_cat_random import ProxySegEarthSegmentationCatRandom   
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.patches as mpatches
import torch

img_path = 'demo/image/kyoto_33.tif'
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


img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((448, 448))
])(img)

img_tensor = img_tensor.unsqueeze(0).to('cuda')
# img_tensor = torch.rot90(img_tensor, k=2, dims=(2,3))

# ----------- SAM preprocess ------------
sam_mean = [123.675/255.0, 116.28/255.0, 103.53/255.0]   # RGB, 0-1 스케일
sam_std  = [58.395/255.0, 57.12/255.0, 57.375/255.0]
img_sam = transforms.Compose([
    transforms.ToTensor(),                                # (H,W,C)[0-255] → (C,H,W)[0-1]
    transforms.Normalize(sam_mean, sam_std),              # SAM 백본 통계
    transforms.Resize((2048, 2048))                       # SAM ViT-B/L 기본 해상도
])(img)

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
# model = SamProxySegEarthSegmentation(
#     clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
#     vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
#     model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
#     ignore_residual=True,
#     feature_up=True,
#     feature_up_cfg=dict(
#         model_name='jbu_one',
#         model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
#     cls_token_lambda=-0.3,
#     name_path='./configs/my_name.txt',
#     prob_thd=0.1,
#     cls_variant="none",
#     vfm_model="sam"
# )

seg_pred = model.predict(img_tensor, data_samples=None)
# seg_pred = torch.rot90(seg_pred, k=2, dims=(1,2))
#seg_pred = model.predict(img_tensor, img_sam, data_samples=None)
#seg_mask = seg_pred.data.cpu().numpy().squeeze(0)
seg_mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(img)
# ax[0].axis('off')
# ax[1].imshow(seg_mask, cmap='viridis')
# ax[1].axis('off')
# plt.tight_layout()
# # plt.show()
# plt.savefig('visualize/prediction/kyoto33_pred.png', bbox_inches='tight')
# plt.close(fig)

# fig, ax = plt.subplots(figsize=(8, 8))
# resized_img = img.resize((448, 448), resample=Image.Resampling.BILINEAR)
# ax.imshow(resized_img)
# cmap = plt.get_cmap('tab20', len(name_list))
# ax.imshow(seg_mask, cmap=cmap, alpha=0.5, vmin=0, vmax=len(name_list)-1)

# import random
# random.seed(42)  # 재현성 위해 시드 고정 (선택)

# for class_idx, class_name in enumerate(name_list):
#     ys, xs = np.where(seg_mask == class_idx)
#     if ys.size == 0:
#         continue
#     # 픽셀 좌표 중 하나를 무작위 선택
#     i = random.randrange(len(ys))
#     y_s, x_s = ys[i], xs[i]
#     ax.text(
#         x_s, y_s,                     # 텍스트 위치
#         class_name.split(',')[0],     # 첫 번째 이름만
#         color='white',
#         fontsize=8,
#         weight='bold',
#         ha='center',
#         va='center',
#         bbox=dict(facecolor='black', alpha=0.5, pad=1, lw=0)
#     )

# ax.axis('off')
# plt.tight_layout()
# plt.savefig('visualize/kyoto33_pred_overlay.png', bbox_inches='tight')
# plt.close(fig)

# === 6. Define the OpenEarthMap 9-class color map ===
class_info = {
    0: ("Background",        "#000000"),  # black
    1: ("Bareland",          "#800000"),
    2: ("Rangeland (grass)", "#00FF24"),
    3: ("Developed space (pavement)",   "#949494"),
    4: ("Road",              "#FFFFFF"),
    5: ("Tree",              "#226126"),
    6: ("Water",             "#0045FF"),
    7: ("Agriculture land (cropland)",  "#4BB549"),
    8: ("Building",          "#DE1F07"),
}
colors = [mcolors.hex2color(class_info[i][1]) for i in range(9)]
cmap_9 = mcolors.ListedColormap(colors)
norm_9 = mcolors.BoundaryNorm(np.arange(10) - 0.5, cmap_9.N)

# === 7. Functions to save visualizations ===
def save_mask_with_legend(mask, filename, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mask, cmap=cmap_9, norm=norm_9)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(9),
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([class_info[i][0] for i in range(9)])
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

# def save_overlay(original_img, mask, filename):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     resized = original_img.resize(mask.shape[::-1],
#                                   resample=Image.Resampling.BILINEAR)
#     ax.imshow(resized)
#     ax.imshow(mask, cmap=cmap_9, norm=norm_9, alpha=0.25)
#     ax.axis("off")
#     plt.tight_layout()
#     plt.savefig(filename, bbox_inches="tight", dpi=300)
#     plt.close(fig)
    
# === 8. Save the outputs ===
if model.feature_up:
    pred_path = Path("visualize/cat") / f"0824_{base_name}_pred.png"
    overlay_path = Path("visualize/cat") / f"0822_{base_name}_pred_overlay.png"
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
# save_overlay(
#     img,
#     seg_mask,
#     filename=str(overlay_path)
# )