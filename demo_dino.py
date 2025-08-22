from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.patches as mpatches
import torch

img_path = 'demo/image/dolnoslaskie_27.tif'
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

model = SegEarthSegmentation(
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

top, left = 224, 224
height, width = 224, 224
crop = img_tensor[:, :, top:top+height, left:left+width]  # [B, C, 224, 224]

save_path = Path("visualize/sam") / f"{base_name}_dino"

for (y_pix, x_pix) in [(0, 0)]: #[(40, 80), (40, 180), (80, 60), (80, 150), (120, 70), (120, 190), (180, 100), (180, 180), (200, 20), (200, 200)]:
    model.viz_patch_attention(crop, y_pix=y_pix, x_pix = x_pix, save_dir=save_path)
# for q in [30, 60, 90, 120, 150, 180, 210]:
#     model.viz_patch_attention(crop, idx_q=q, save_dir=save_path)
# model.viz_patch_attention(crop, idx_q=264, save_dir=save_path)


# -----------------------------
feat1 = model.vfm_forward(crop1)['ex_feats']
feat2 = model.vfm_forward(crop2)['ex_feats']

idx_q = 1

similarity = torch.einsum("b c m, b c n -> b m n", feat1, feat2)
similarity[similarity < 0.3] = float('-inf')
attn = F.softmax(similarity, dim=-1)

B, N, _ = attn.shape
hw = int(math.sqrt(N))

img_disp = (crop2[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
        
heat = attn[0][idx_q].cpu().reshape(hw, hw)  # (H', W')
heat_up = heat / np.max(heat)

# ── 1) idx_q → (cx, cy) 픽셀 좌표 ─────────────────────────────
patch_h = imgs.shape[-2] // hw     # = patch_w
patch_w = imgs.shape[-1] // hw
row_q   = idx_q // hw
col_q   = idx_q %  hw
cy      = row_q * patch_h + patch_h // 2   # y (row)  중심
cx      = col_q * patch_w + patch_w // 2   # x (col)  중심

import matplotlib.pyplot as plt
fig, ax = plt.subplots(
    1, 2, figsize=((imgs.shape[-1]*2)/dpi, imgs.shape[-2]/dpi), dpi=dpi)

# 왼쪽 : 원본 + 표시점
ax[0].imshow(img_disp)
ax[0].scatter(cx, cy, s=8, c='red', marker='x', linewidths=1)
ax[0].set_title(f"Query patch ({y_pix}, {x_pix})", fontsize=5)
ax[0].axis("off")

# 오른쪽 : 히트맵 overlay
ax[1].imshow(img_disp)
ax[1].imshow(heat_up, alpha=0.7, cmap="jet", vmin=0, vmax=1)
ax[1].set_title("Attention heat‑map", fontsize=5)
ax[1].axis("off")

# ── 3) 저장 ────────────────────────────────────────────────
file_name = f"{prefix}_{y_pix}_{x_pix}.png"
file_path = os.path.join(dir_path, file_name)
plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
plt.close(fig)

print(f"Saved attention maps to '{file_path}'")

# -------------------------------

