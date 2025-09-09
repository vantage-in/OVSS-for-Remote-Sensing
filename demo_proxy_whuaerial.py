import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
from proxy_segearth_segmentor import ProxySegEarthSegmentation
from proxy_segearth_segmentor_cat_random import ProxySegEarthSegmentationCatRandom
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.patches as mpatches
import os

# --- Control Panel ---
# Set to True to process and save the corresponding label masks
# Set to False to only save the predictions
PROCESS_LABELS = False 

# --- 1. Define Paths ---
image_dir = Path('demo/whu_aerial/image')
label_dir = Path('data/WHU-BD/val/label_cvt')
output_dir = Path('visualize/whu_aerial')

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# --- 2. Setup Model and Class Configuration ---
name_list = ['background', 'building']
class_info = {
    0: ("background", "#000000"), 
    1: ("building", "#FFFFFF")
}
name_file_path = './configs/my_name.txt'

# Write class names to a file for the model
with open(name_file_path, 'w') as writers:
    for i, name in enumerate(name_list):
        writers.write(name)
        if i < len(name_list) - 1:
            writers.write('\n')

# --- 3. Initialize the Segmentation Model ---
# This is done only once to save time and resources.
print("Initializing the segmentation model...")
# model = ProxySegEarthSegmentationCatRandom(
#     clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
#     vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
#     model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
#     ignore_residual=True,
#     feature_up=True,
#     feature_up_cfg=dict(
#         model_name='jbu_one',
#         model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
#     cls_token_lambda=-0.3,
#     name_path=name_file_path,
#     prob_thd=0.65,
#     cls_variant="none",
#     vfm_model="dino"
# )
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
    name_path=name_file_path,
    prob_thd=0.6,
    cls_variant="none"
)
print("Model initialized successfully.")

# --- 4. Define Image Transformations ---
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((448, 448))
])

# --- 5. Define Visualization Colors and Functions ---
colors = [mcolors.hex2color(class_info[i][1]) for i in range(len(class_info))]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(len(class_info) + 1) - 0.5, cmap.N)

def save_mask_with_legend(mask, filename, title):
    """Saves a segmentation mask with a color bar legend."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mask, cmap=cmap, norm=norm)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(class_info)),
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([class_info[i][0] for i in range(len(class_info))])
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)

# --- 6. Main Processing Loop ---
# Get a list of all image files in the directory
image_files = sorted(list(image_dir.glob('*.png'))) # Use glob to find all .png files

if not image_files:
    print(f"Error: No images found in the directory: {image_dir}")
else:
    print(f"Found {len(image_files)} images to process.")

for img_path in image_files:
    base_name = img_path.stem
    print(f"Processing: {img_path.name}...")

    # --- 6.1. Process and Predict Image ---
    try:
        img = Image.open(img_path)
        img_tensor = img_transform(img).unsqueeze(0).to('cuda')
        
        # Get model prediction
        seg_pred = model.predict(img_tensor, data_samples=None)
        seg_mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

        # Save the prediction mask
        pred_path = output_dir / f"{base_name}_pred_segearth.png"
        save_mask_with_legend(
            seg_mask,
            filename=str(pred_path),
            title=f"Prediction - {base_name}"
        )
        print(f"  -> Saved prediction to {pred_path}")

    except Exception as e:
        print(f"  -> Failed to process prediction for {img_path.name}. Error: {e}")
        continue # Skip to the next image

    # --- 6.2. (Optional) Process Corresponding Label ---
    if PROCESS_LABELS:
        label_path = label_dir / f"{base_name}.png"

        if label_path.exists():
            try:
                label_img = Image.open(label_path)
                label_array = np.array(label_img)

                # Save the label mask
                label_output_path = output_dir / f"{base_name}_label.png"
                save_mask_with_legend(
                    label_array,
                    filename=str(label_output_path),
                    title=f"Label - {base_name}"
                )
                print(f"  -> Saved label to {label_output_path}")

            except Exception as e:
                print(f"  -> Failed to process label file {label_path.name}. Error: {e}")
        else:
            print(f"  -> Warning: Label file not found for {img_path.name} at {label_path}")

print("\nProcessing complete.")

# from PIL import Image
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from segearth_segmentor import SegEarthSegmentation
# from proxy_segearth_segmentor import ProxySegEarthSegmentation
# from proxy_segearth_segmentor_cat_random import ProxySegEarthSegmentationCatRandom
# import numpy as np
# import matplotlib.colors as mcolors
# from pathlib import Path
# import matplotlib.patches as mpatches
# import torch

# img_path = 'demo/building/image/val_156.png'
# img = Image.open(img_path)
# base_name = Path(img_path).stem 


# name_list = ['background', 'building']

# with open('./configs/my_name.txt', 'w') as writers:
#     for i in range(len(name_list)):
#         if i == len(name_list)-1:
#             writers.write(name_list[i])
#         else:
#             writers.write(name_list[i] + '\n')
# writers.close()

# img_tensor = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
#     transforms.Resize((448, 448))
# ])(img)

# img_tensor = img_tensor.unsqueeze(0).to('cuda')

# model = ProxySegEarthSegmentationCatRandom(
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
#     vfm_model="dino"
# )

# seg_pred = model.predict(img_tensor, data_samples=None)
# seg_mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

# class_info = {
#     0: ("background", "#000000"), 
#     1: ("building", "#FFFFFF")
# }
# colors = [mcolors.hex2color(class_info[i][1]) for i in range(len(class_info))]
# cmap = mcolors.ListedColormap(colors)
# norm = mcolors.BoundaryNorm(np.arange(len(class_info) + 1) - 0.5, cmap.N)

# # === 7. Functions to save visualizations ===
# def save_mask_with_legend(mask, filename, title):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     im = ax.imshow(mask, cmap=cmap, norm=norm)
#     ax.axis("off")
#     ax.set_title(title, fontsize=14)
#     cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(class_info)),
#                         fraction=0.046, pad=0.04)
#     cbar.ax.set_yticklabels([class_info[i][0] for i in range(len(class_info))])
#     cbar.ax.tick_params(labelsize=8)
#     plt.tight_layout()
#     plt.savefig(filename, bbox_inches="tight", dpi=300)
#     plt.close(fig)

# def save_overlay_with_legend(img, seg_mask, filename, name_list):
#     # 1) 준비
#     fig, ax = plt.subplots(figsize=(8, 8))
#     # 2) 원본 리사이즈
#     resized_img = img.resize((448, 448), resample=Image.Resampling.BILINEAR)
#     ax.imshow(resized_img)
#     # 3) 분할 colormap
#     cmap = plt.get_cmap('tab20', len(name_list))
#     ax.imshow(seg_mask,
#               cmap=cmap,
#               alpha=0.5,
#               vmin=0, vmax=len(name_list)-1)

#     # 5) legend 추가
#     patches = []
#     for idx, full_name in enumerate(name_list):
#         label = full_name.split(',')[0]
#         color = cmap(idx)
#         patches.append(mpatches.Patch(color=color, label=label))
#     ax.legend(
#         handles=patches,
#         bbox_to_anchor=(1.05, 1.0),
#         loc='upper left',
#         fontsize=8,
#         frameon=False
#     )

#     # 6) 마무리 저장
#     ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(filename, bbox_inches='tight', dpi=300)
#     plt.close(fig)
    
# # === 8. Save the outputs ===
# pred_path = Path("visualize/building") / f"{base_name}_pred.png"
# overlay_path = Path("visualize/building") / f"{base_name}_pred_overlay.png"

# save_mask_with_legend(
#     seg_mask,
#     filename=str(pred_path),
#     title=f"WHU-Aerial - {base_name}"
# )
# # save_overlay_with_legend(
# #     img,
# #     seg_mask,
# #     filename=str(overlay_path),
# #     name_list=name_list
# # )


# # === 사용자 입력 ===
# label_path = "data/WHU-BD/val/label_cvt/val_156.png"  # 단일 채널 label mask 경로

# class_info = {
#     0: ("background", "#000000"), 
#     1: ("building", "#FFFFFF")
# }

# colors = [mcolors.hex2color(class_info[i][1]) for i in range(len(class_info))]
# cmap = mcolors.ListedColormap(colors)
# norm = mcolors.BoundaryNorm(np.arange(len(class_info) + 1) - 0.5, cmap.N)

# # === label mask 불러오기 ===
# label_img = Image.open(label_path)
# label_array = np.array(label_img)

# # === 파일 이름 추출 ===
# filename = Path(label_path).stem

# # === 시각화 ===
# fig, ax = plt.subplots(figsize=(8, 8))
# im = ax.imshow(label_array, cmap=cmap, norm=norm)
# ax.axis('off')
# ax.set_title(f"OpenEarthMap - {filename}", fontsize=14)

# # colorbar 추가 (이미지 세로 높이 맞춤)
# cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(class_info)), fraction=0.046, pad=0.04)
# cbar.ax.set_yticklabels([class_info[i][0] for i in range(len(class_info))])
# cbar.ax.tick_params(labelsize=8)

# plt.tight_layout()
# pth = f'./visualize/building/{filename}_label.png'
# plt.savefig(pth, bbox_inches='tight')
