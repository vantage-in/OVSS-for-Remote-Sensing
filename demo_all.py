import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path

# --- Import Model Classes ---
from segearth_segmentor import SegEarthSegmentation
from proxy_segearth_segmentor_cat_random import ProxySegEarthSegmentationCatRandom

# --- 1. Configuration ---
# Choose the model to run: 'ours', 'segearth', or 'both'
MODEL_CHOICE = 'segearth' 

# Set directories for input
IMAGE_DIR = Path('demo/image')
# Updated LABEL_DIR to the new path
LABEL_DIR = Path('data/OpenEarthMap/ann_dir/val/')
BASE_OUTPUT_DIR = Path('present/openearthmap')

# Option to process label files. Set to False to disable.
PROCESS_LABELS = False

# --- 2. Define Color Map for Visualization ---
class_info = {
    0: ("Background", "#000000"),
    1: ("Bareland", "#800000"),
    2: ("Rangeland (grass)", "#00FF24"),
    3: ("Developed space (pavement)", "#949494"),
    4: ("Road", "#FFFFFF"),
    5: ("Tree", "#226126"),
    6: ("Water", "#0045FF"),
    7: ("Agriculture land (cropland)", "#4BB549"),
    8: ("Building", "#DE1F07"),
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

def run_inference(model_name, output_dir_name):
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
            prob_thd=0.1, cls_variant="none", vfm_model="dino"
        )
    elif model_name == 'segearth':
        model = SegEarthSegmentation(
            clip_type='CLIP', vit_type='ViT-B/16', model_type='SegEarth',
            ignore_residual=True, feature_up=True,
            feature_up_cfg=dict(
                model_name='jbu_one',
                model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
            cls_token_lambda=-0.3, name_path='./configs/my_name.txt',
            prob_thd=0.1, cls_variant="none"
        )
    model.to('cuda')
    
    # 3. Process all images
    image_files = list(IMAGE_DIR.glob('*.tif')) + list(IMAGE_DIR.glob('*.png')) + list(IMAGE_DIR.glob('*.jpg'))
    for img_path in image_files:
        print(f"  Predicting for {img_path.name}...")
        base_name = img_path.stem
        img = Image.open(img_path).convert("RGB")

        img_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            transforms.Resize((448, 448))
        ])(img)
        img_tensor = img_tensor.unsqueeze(0).to('cuda')

        seg_pred = model.predict(img_tensor, data_samples=None)
        seg_mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

        pred_output_path = output_dir / f"{base_name}_{model_name}.png"
        save_plain_mask(seg_mask, pred_output_path)

# --- 4. Main Execution Logic ---
# Generate the model configuration file
name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
             'tree,forest', 'water,river', 'cropland', 'building,roof,house']
with open('./configs/my_name.txt', 'w') as writer:
    for item in name_list:
        writer.write(item + '\n')

# Step 1: Process all labels once (if enabled)
if PROCESS_LABELS:
    print("\n--- Processing Labels ---")
    output_label_dir = BASE_OUTPUT_DIR / 'label'
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(IMAGE_DIR.glob('*.tif')) + list(IMAGE_DIR.glob('*.png')) + list(IMAGE_DIR.glob('*.jpg'))
    for img_path in image_files:
        label_path = LABEL_DIR / img_path.name
        if label_path.is_file():
            print(f"  Saving label for {img_path.name}...")
            label_img = Image.open(label_path)
            label_mask = np.array(label_img)
            label_output_path = output_label_dir / f"{img_path.stem}_label.png"
            save_plain_mask(label_mask, label_output_path)
        else:
            print(f"  -> Warning: No corresponding label found at {label_path}")

# Step 2: Run the selected model(s) for predictions
if MODEL_CHOICE == 'ours':
    run_inference(model_name='ours', output_dir_name='ours')
elif MODEL_CHOICE == 'segearth':
    run_inference(model_name='segearth', output_dir_name='segearth')
elif MODEL_CHOICE == 'both':
    run_inference(model_name='ours', output_dir_name='ours')
    run_inference(model_name='segearth', output_dir_name='segearth')
else:
    raise ValueError("Invalid MODEL_CHOICE. Please choose 'ours', 'segearth', or 'both'.")

print("\nAll processing complete.")