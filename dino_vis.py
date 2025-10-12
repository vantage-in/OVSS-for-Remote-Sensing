import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2
from torchvision import transforms
from pathlib import Path
import os

# --- 1. 설정 변수 ---
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# 사용자 로컬 이미지 파일 경로로 변경하세요.
IMAGE_PATH = '/home/icl_intern1/SegEarth-OV/data/OpenEarthMap/img_dir/val/baybay_53.tif'
IMAGE_PATH = '/home/icl_intern1/SegEarth-OV/data/VDD/test/src/DJI_10507.JPG'
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# --- 2. 모델 및 전처리기 준비 ---
def setup_model_and_transforms():
    """DINO 모델과 이미지 전처리기를 로드하고 설정합니다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    return model, transform, device

def get_dino_patch_embeddings(image_tensor, model, device):
    """주어진 이미지 텐서에서 DINO 패치 임베딩을 추출합니다."""
    with torch.no_grad():
        embeddings = model.get_intermediate_layers(image_tensor.to(device))[0]
        return embeddings[0, 1:, :]

# --- 헬퍼 함수 추가 ---
def resize_with_aspect_ratio(img, target_dim):
    """이미지 비율을 유지하며 긴 변을 target_dim에 맞게 리사이즈합니다."""
    w, h = img.size
    scale = target_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)

def resize_and_pad(img, target_size, fill_color=(0, 0, 0)):
    """비율을 유지하며 리사이즈하고, 남는 공간을 패딩하여 정방형으로 만듭니다."""
    resized_img = img.copy()
    resized_img.thumbnail(target_size, Image.LANCZOS)
    new_img = Image.new("RGB", target_size, fill_color)
    paste_position = ((target_size[0] - resized_img.width) // 2, (target_size[1] - resized_img.height) // 2)
    new_img.paste(resized_img, paste_position)
    return new_img, paste_position

# --- 3. 핵심 시각화 함수 ---
def visualize_embedding_similarity(model, transform, device, img_path, source_crop_loc, k_index, comparison_target_loc='global'):
    try:
        original_img = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: The file was not found at {img_path}")
        return None

    # --- Step 1: 이미지 형태에 따른 동적 Crop 위치 계산 ---
    is_square = original_img.width == original_img.height
    crop_size = 224
    stride = 112

    # Local 임베딩 추출 기준이 되는 '패딩 없는' 이미지
    img_local_base = resize_with_aspect_ratio(original_img, 448)
    W, H = img_local_base.size

    # 시각화용 '패딩된' 이미지
    img_448_padded, paste_pos = resize_and_pad(original_img, (448, 448))

    # 이미지 크기에 따라 동적으로 crop 위치 계산
    y_starts = []
    if is_square:
        y_starts = [0, stride, 448 - crop_size]
    else:
        y = 0
        while y + crop_size <= H:
            y_starts.append(y)
            if y + crop_size == H: break
            y += stride
        if H > crop_size and (len(y_starts) == 0 or y_starts[-1] != H - crop_size):
            y_starts.append(H - crop_size)
        y_starts = sorted(list(set(y_starts)))

    x_starts = [0, (W - crop_size) // 2, W - crop_size]
    v_names, h_names = ['top', 'middle', 'bottom'], ['_left', '_center', '_right']
    
    active_locs = [v_names[i] + h_names[j] for i in range(len(y_starts)) for j in range(len(x_starts))]
    if source_crop_loc not in active_locs or (comparison_target_loc != 'global' and comparison_target_loc not in active_locs):
        print(f"Error: Invalid location specified. Available: {active_locs}")
        return None

    active_loc_to_idx = {loc: i for i, loc in enumerate(active_locs)}

    # --- Step 2: 소스 패치 임베딩 추출 ---
    source_idx = active_loc_to_idx[source_crop_loc]
    source_v_idx, source_h_idx = source_idx // len(x_starts), source_idx % len(x_starts)
    source_crop_x, source_crop_y = x_starts[source_h_idx], y_starts[source_v_idx]
    
    source_crop_img = img_local_base.crop((source_crop_x, source_crop_y, source_crop_x + crop_size, source_crop_y + crop_size))
    source_tensor = transform(source_crop_img).unsqueeze(0)
    source_embeddings = get_dino_patch_embeddings(source_tensor, model, device)
    
    if not (0 <= k_index < source_embeddings.shape[0]):
        raise ValueError(f"k_index must be between 0 and {source_embeddings.shape[0]-1}")
    target_embedding = source_embeddings[k_index].unsqueeze(0)

    # --- Step 3: 비교 대상 이미지 및 임베딩 준비 ---
    if comparison_target_loc == 'global':
        target_img, _ = resize_and_pad(original_img, (224, 224))
        target_tensor = transform(target_img).unsqueeze(0)
        embeddings_target = get_dino_patch_embeddings(target_tensor, model, device)
    else: # 타일 레벨 비교
        target_idx = active_loc_to_idx[comparison_target_loc]
        target_v_idx, target_h_idx = target_idx // len(x_starts), target_idx % len(x_starts)
        target_crop_x, target_crop_y = x_starts[target_h_idx], y_starts[target_v_idx]

        # 모델 입력용: 패딩 없는 이미지에서 crop
        target_crop_img_for_model = img_local_base.crop((target_crop_x, target_crop_y, target_crop_x + crop_size, target_crop_y + crop_size))
        target_tensor = transform(target_crop_img_for_model).unsqueeze(0)
        embeddings_target = get_dino_patch_embeddings(target_tensor, model, device)
        
        # 시각화용: 패딩된 이미지에서 crop (정방형 유지를 위해)
        viz_crop_x = paste_pos[0] + target_crop_x
        viz_crop_y = paste_pos[1] + target_crop_y
        target_img = img_448_padded.crop((viz_crop_x, viz_crop_y, viz_crop_x + crop_size, viz_crop_y + crop_size))

    # --- Step 4: 코사인 유사도 맵 생성 ---
    cosine_sim = F.cosine_similarity(target_embedding, embeddings_target, dim=1)
    sim_map = cosine_sim.reshape(28, 28).cpu().numpy()

    # --- Step 5: 시각화 이미지 생성 ---
    sim_map_resized = cv2.resize(sim_map, (224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap_raw = cv2.applyColorMap(np.uint8(255 * np.clip(sim_map_resized, 0, 0.5) / 0.5), cv2.COLORMAP_JET)
    similarity_map_img = Image.fromarray(cv2.cvtColor(heatmap_raw, cv2.COLOR_BGR2RGB))
    overlay_img = Image.blend(target_img, similarity_map_img, alpha=0.5)

    # 소스 패치 위치를 '패딩 없는' 이미지에 그리기
    patch_size, grid_size = 8, 28
    patch_row, patch_col = k_index // grid_size, k_index % grid_size
    local_x, local_y = patch_col * patch_size, patch_row * patch_size
    
    global_x = source_crop_x + local_x
    global_y = source_crop_y + local_y
    
    source_img_with_box = img_local_base.copy()
    draw = ImageDraw.Draw(source_img_with_box)
    draw.rectangle([global_x, global_y, global_x + patch_size, global_y + patch_size], outline='red', width=2)
    
    return source_img_with_box, target_img, similarity_map_img, overlay_img, active_locs

# --- 4. 메인 실행 부분 ---
if __name__ == "__main__":
    if not Path(IMAGE_PATH).exists():
        print(f"!! ERROR: Image file not found at '{IMAGE_PATH}'")
    else:
        dino_model, dino_transform, dino_device = setup_model_and_transforms()
        output_dir = Path("visualize/dino_scale_dynamic")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Demo a: Global 이미지와 비교 ---
        print("\n--- Running Demo (a): Comparing with Global Image ---")
        SOURCE_LOC_A = 'middle_left'
        K_INDEX_A = 434 
        TARGET_LOC_A = 'global'
        
        results_a = visualize_embedding_similarity(
            dino_model, dino_transform, dino_device, IMAGE_PATH,
            SOURCE_LOC_A, K_INDEX_A, TARGET_LOC_A
        )

        if results_a:
            source_img, target_img, sim_map, overlay, active_locs = results_a
            print(f"Image processed. Available local positions: {active_locs}")
            prefix = f"{Path(IMAGE_PATH).stem}_source_{SOURCE_LOC_A}_{K_INDEX_A}_target_{TARGET_LOC_A}"
            source_img.save(output_dir / f"{prefix}_1_source_patch_location.png")
            target_img.save(output_dir / f"{prefix}_2_target_image.png")
            sim_map.save(output_dir / f"{prefix}_3_similarity_map.png")
            overlay.save(output_dir / f"{prefix}_4_overlay.png")
            print(f"Successfully saved 4 images for Demo (a) to '{output_dir}'")
        else:
            print("Could not generate visualization for Demo (a).")

        # --- Demo b: Tile Level 이미지와 비교 ---
        print("\n--- Running Demo (b): Comparing with Another Tile ---")
        SOURCE_LOC_B = 'top_left'
        K_INDEX_B = 189
        TARGET_LOC_B = 'middle_right'

        results_b = visualize_embedding_similarity(
            dino_model, dino_transform, dino_device, IMAGE_PATH,
            SOURCE_LOC_B, K_INDEX_B, TARGET_LOC_B
        )

        if results_b:
            source_img, target_img, sim_map, overlay, _ = results_b
            prefix = f"{Path(IMAGE_PATH).stem}_source_{SOURCE_LOC_B}_{K_INDEX_B}_target_{TARGET_LOC_B}"
            source_img.save(output_dir / f"{prefix}_1_source_patch_location.png")
            target_img.save(output_dir / f"{prefix}_2_target_image.png")
            sim_map.save(output_dir / f"{prefix}_3_similarity_map.png")
            overlay.save(output_dir / f"{prefix}_4_overlay.png")
            print(f"Successfully saved 4 images for Demo (b) to '{output_dir}'")
        else:
            print("Could not generate visualization for Demo (b).")