import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
import random
import shutil

# --- Import Model Classes ---
from segearth_segmentor import SegEarthSegmentation
from proxy_segearth_segmentor_cat_random import ProxySegEarthSegmentationCatRandom

# --- 1. Configuration ---
# Choose the model to run: 'ours', 'segearth', or 'both'
MODEL_CHOICE = 'both' 

# Set directories for input
IMAGE_DIR = Path('data/GlobalRoadSet_Val/DeepGlobe_test_1530/image_cvt')
# Updated LABEL_DIR to the new path
LABEL_DIR = Path('data/GlobalRoadSet_Val/DeepGlobe_test_1530/label_cvt')
BASE_OUTPUT_DIR = Path('present/deepglobe')

# Option to process label files. Set to False to disable.
PROCESS_LABELS = True

# Num of sampling
NUM_SAMPLES = 100

# --- 2. Define Color Map for Visualization ---
class_info = {
    0: ("background", "#000000"),                # 네이비
    1: ("road", "#FFFFFF")
}
colors = [mcolors.hex2color(class_info[i][1]) for i in range(len(class_info))]
cmap_oem = mcolors.ListedColormap(colors)
norm_oem = mcolors.BoundaryNorm(np.arange(len(class_info) + 1) - 0.5, cmap_oem.N)

# --- 3. Define Core Functions ---
def save_plain_mask(mask, output_path):
    """Saves a segmentation mask as an image with no border, title, or legend."""
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap=cmap_oem, norm=norm_oem)
    ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)

def run_inference(model_name, output_dir_name, image_files):
    """Initializes a model and runs inference, saving only the predictions."""
    print(f"\n--- Running Inference for: {model_name} ---")
    print(f"--- Outputting predictions to: {output_dir_name} ---")
    
    # 1. Setup Prediction Output Directory
    output_dir = BASE_OUTPUT_DIR / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
        
    # 2. Initialize Model
    if model_name == 'ours':
        model = ProxySegEarthSegmentationCatRandom(
            clip_type='CLIP', vit_type='ViT-B/16', model_type='SegEarth',
            ignore_residual=True, feature_up=True,
            feature_up_cfg=dict(
                model_name='jbu_one',
                model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
            cls_token_lambda=-0.3, name_path='./configs/my_name.txt',
            prob_thd=0.75, cls_variant="none", vfm_model="dino"
        )
    elif model_name == 'segearth':
        model = SegEarthSegmentation(
            clip_type='CLIP', vit_type='ViT-B/16', model_type='SegEarth',
            ignore_residual=True, feature_up=True,
            feature_up_cfg=dict(
                model_name='jbu_one',
                model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
            cls_token_lambda=-0.3, name_path='./configs/my_name.txt',
            prob_thd=0.7, cls_variant="none"
        )
    model.to('cuda')
    
    # 3. Process all images
    # image_files = list(IMAGE_DIR.glob('*.tif')) + list(IMAGE_DIR.glob('*.png')) + list(IMAGE_DIR.glob('*.jpg'))
    for img_path in image_files:
        print(f"  Predicting for {img_path.name}...")
        base_name = img_path.stem
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        if w > h:
            new_w = 448
            new_h = int(h * (448 / w))
        else:
            new_h = 448
            new_w = int(w * (448 / h))

        img_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            transforms.Resize((new_h, new_w))
        ])(img)
        img_tensor = img_tensor.unsqueeze(0).to('cuda')

        seg_pred = model.predict(img_tensor, data_samples=None)
        seg_mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

        pred_output_path = output_dir / f"{base_name}_{model_name}.png"
        save_plain_mask(seg_mask, pred_output_path)

def save_legend_image(class_info, output_path):
    """Saves a legend image based on the class_info dictionary."""
    print(f"\n--- Creating Legend Image ---")
    
    # 1. 범례에 사용할 색상 패치와 클래스 이름 리스트를 만듭니다.
    patches = [mpatches.Patch(color=info[1], label=f"{key}: {info[0]}")
               for key, info in class_info.items()]
    
    # 2. 범례만 포함하는 그림(figure)을 생성합니다.
    # 범례의 크기에 맞춰 그림 크기를 조절할 수 있습니다 (figsize).
    fig, ax = plt.subplots(figsize=(4, 2.5))
    
    # 3. 그림의 축을 끄고 범례를 추가합니다.
    ax.axis('off')
    fig.legend(handles=patches, loc='center', fontsize='large', frameon=False)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 4. 여백 없이 깔끔하게 이미지 파일로 저장합니다.
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f"Legend saved to {output_path}")

# --- 4. Main Execution Logic ---
# Generate the model configuration file
name_list = ['background', 'road']
with open('./configs/my_name.txt', 'w') as writer:
    for item in name_list:
        writer.write(item + '\n')

if PROCESS_LABELS:
    legend_output_path = BASE_OUTPUT_DIR / "legend.png"
    save_legend_image(class_info, legend_output_path)

print("\n--- Preparing and Sampling File List ---")
all_image_files = sorted(list(IMAGE_DIR.glob('*.tif')) + list(IMAGE_DIR.glob('*.png')) + list(IMAGE_DIR.glob('*.jpg'))  + list(IMAGE_DIR.glob('*.JPG')))

if len(all_image_files) > NUM_SAMPLES:
    print(f"Randomly sampling {NUM_SAMPLES} images from {len(all_image_files)} total files.")
    sampled_image_files = random.sample(all_image_files, NUM_SAMPLES)
else:
    print(f"Fewer than {NUM_SAMPLES} images found. Using all {len(all_image_files)} files.")
    sampled_image_files = all_image_files

print("\n--- Copying Sampled Images ---")
output_image_dir = BASE_OUTPUT_DIR / 'image'
output_image_dir.mkdir(parents=True, exist_ok=True) # 복사할 폴더 생성

for img_path in sampled_image_files:
    # shutil.copy(원본 경로, 대상 경로)
    shutil.copy(img_path, output_image_dir / img_path.name)
print(f"Copied {len(sampled_image_files)} images to {output_image_dir}")

# Step 1: Process all labels once (if enabled)
if PROCESS_LABELS:
    print("\n--- Processing Labels ---")
    output_label_dir = BASE_OUTPUT_DIR / 'label'
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # image_files = list(IMAGE_DIR.glob('*.tif')) + list(IMAGE_DIR.glob('*.png')) + list(IMAGE_DIR.glob('*.jpg'))
    # for img_path in image_files:
    for img_path in sampled_image_files:
        base_name = img_path.stem
        found_labels = list(LABEL_DIR.glob(f"{base_name}.*"))
        
        if found_labels:
            # 여러 개가 찾아지면 첫 번째 파일을 사용합니다.
            label_path = found_labels[0]
            print(f"  Saving label for {img_path.name} (found as {label_path.name})...")
            
            label_img = Image.open(label_path)
            label_mask = np.array(label_img)
            # 출력 파일명은 이미지의 base_name을 기준으로 통일합니다.
            label_output_path = output_label_dir / f"{base_name}_label.png"
            save_plain_mask(label_mask, label_output_path)
        else:
            # 일치하는 라벨 파일이 없는 경우 경고 메시지를 출력합니다.
            print(f"  -> Warning: No corresponding label found for {img_path.name} in {LABEL_DIR}")

# Step 2: Run the selected model(s) for predictions
if MODEL_CHOICE == 'ours':
    run_inference(model_name='ours', output_dir_name='ours', image_files=sampled_image_files)
elif MODEL_CHOICE == 'segearth':
    run_inference(model_name='segearth', output_dir_name='segearth', image_files=sampled_image_files)
elif MODEL_CHOICE == 'both':
    run_inference(model_name='ours', output_dir_name='ours', image_files=sampled_image_files)
    run_inference(model_name='segearth', output_dir_name='segearth', image_files=sampled_image_files)
else:
    raise ValueError("Invalid MODEL_CHOICE. Please choose 'ours', 'segearth', or 'both'.")

print("\nAll processing complete.")