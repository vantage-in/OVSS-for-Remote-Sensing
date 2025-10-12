import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2
from pathlib import Path
import os
import os.path as osp

# --- 1. 설정 변수 ---
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# 사용자 로컬 이미지 파일 경로로 변경하세요.
IMAGE_PATH = '/home/icl_intern1/SegEarth-OV/data/OpenEarthMap/img_dir/val/baybay_53.tif'
IMAGE_PATH = '/home/icl_intern1/SegEarth-OV/data/VDD/test/src/DJI_10507.JPG'
# 미리 추출된 DINOv3 임베딩 .pt 파일이 있는 디렉토리 경로
VFM_EMBEDDING_DIR = 'dinov3_features_448/OpenEarthMap'
VFM_EMBEDDING_DIR = 'dinov3_features_448/VDD'
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# --- 2. 헬퍼 함수 ---

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
    paste_position = (
        (target_size[0] - resized_img.width) // 2,
        (target_size[1] - resized_img.height) // 2
    )
    new_img.paste(resized_img, paste_position)
    return new_img, paste_position

# --- 3. 핵심 시각화 함수 ---
def visualize_embedding_similarity(device, img_path, vfm_embedding_dir, source_crop_loc, k_index, comparison_target_loc='global'):
    try:
        original_img = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: The image file was not found at {img_path}")
        return None
    
    # --- Step 1: DINOv3 임베딩 로드 ---
    filename_without_ext = Path(img_path).stem
    vfm_embedding_path = osp.join(vfm_embedding_dir, f"{filename_without_ext}.pt")
    try:
        vfm_embeddings = torch.load(vfm_embedding_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: The embedding file was not found at {vfm_embedding_path}")
        return None
    
    global_feats = vfm_embeddings['global_dino_feats_flat'].to(dtype=torch.float32)
    all_robust_feats = vfm_embeddings['all_robust_dino_feats'].to(dtype=torch.float32)
    
    # --- Step 2: 이미지 형태에 따른 동적 Crop 위치 계산 ---
    is_square = original_img.width == original_img.height
    crop_size = 224
    stride = 112
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ 수정된 부분 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # 1. Local 임베딩 추출 기준이 되는 '패딩 없는' 이미지
    img_local_base = resize_with_aspect_ratio(original_img, 448)
    W, H = img_local_base.size

    # 2. Global 임베딩 및 시각화용 '패딩된' 이미지
    img_448_padded, paste_pos = resize_and_pad(original_img, (448, 448))
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    active_locs = []
    
    if is_square:
        y_starts = [0, stride, 448 - crop_size]
    else:
        y_starts = []
        y = 0
        while y + crop_size <= H:
            y_starts.append(y)
            if y + crop_size == H: break
            y += stride
        if H > crop_size and (len(y_starts) == 0 or y_starts[-1] != H - crop_size):
            y_starts.append(H - crop_size)
        y_starts = sorted(list(set(y_starts)))

    x_starts = [0, (W - crop_size) // 2, W - crop_size]
    v_names = ['top', 'middle', 'bottom']
    h_names = ['_left', '_center', '_right']
    
    for i in range(len(y_starts)):
        for j in range(len(x_starts)):
            active_locs.append(v_names[i] + h_names[j])
    
    num_expected_chunks = all_robust_feats.shape[0] // 784
    if num_expected_chunks != len(active_locs):
        print(f"Warning: Mismatch in embedding chunks. Expected {len(active_locs)} but found {num_expected_chunks}. Using found value.")
        active_locs = active_locs[:num_expected_chunks]

    if source_crop_loc not in active_locs or (comparison_target_loc != 'global' and comparison_target_loc not in active_locs):
        print(f"Error: Invalid location specified. Available locations for this image: {active_locs}")
        return None

    # --- Step 3: 소스 및 타겟 임베딩 선택 ---
    active_loc_to_idx = {loc: i for i, loc in enumerate(active_locs)}
    chunk_size = 784
    
    source_idx = active_loc_to_idx[source_crop_loc]
    source_patch_embeddings = all_robust_feats[source_idx*chunk_size : (source_idx+1)*chunk_size]
    
    if not (0 <= k_index < source_patch_embeddings.shape[0]):
        raise ValueError(f"k_index must be between 0 and {source_patch_embeddings.shape[0]-1}")
    target_embedding = source_patch_embeddings[k_index].unsqueeze(0)

    if comparison_target_loc == 'global':
        target_img = resize_and_pad(original_img, (224, 224))[0]
        embeddings_target = global_feats
    else:
        target_idx = active_loc_to_idx[comparison_target_loc]
        embeddings_target = all_robust_feats[target_idx*chunk_size : (target_idx+1)*chunk_size]
        
        v_idx, h_idx = target_idx // len(x_starts), target_idx % len(x_starts)
        crop_x, crop_y = paste_pos[0] + x_starts[h_idx], paste_pos[1] + y_starts[v_idx]
        target_img = img_448_padded.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))

    # --- Step 4 & 5: 유사도 계산 및 시각화 ---
    cosine_sim = F.cosine_similarity(target_embedding, embeddings_target, dim=1)
    if cosine_sim.numel() == 0: return None
        
    grid_dim = int(np.sqrt(cosine_sim.shape[0]))
    sim_map = cosine_sim.reshape(grid_dim, grid_dim).cpu().numpy()

    sim_map_resized = cv2.resize(sim_map, (224, 224), interpolation=cv2.INTER_LINEAR)
    heatmap_img_raw = cv2.applyColorMap(np.uint8(255 * np.clip(sim_map_resized, 0, 0.5) / 0.5), cv2.COLORMAP_JET)
    similarity_map_img = Image.fromarray(cv2.cvtColor(heatmap_img_raw, cv2.COLOR_BGR2RGB))
    overlay_img = Image.blend(target_img, similarity_map_img, alpha=0.5)

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ 수정된 부분 ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # 소스 패치 위치를 '패딩 없는' 이미지에 그리기
    patch_size = 8
    patch_row, patch_col = k_index // grid_dim, k_index % grid_dim
    local_x, local_y = patch_col * patch_size, patch_row * patch_size
    
    source_v_idx = source_idx // len(x_starts)
    source_h_idx = source_idx % len(x_starts)
    
    # 패딩 없는 이미지(`img_local_base`)에서의 절대 좌표 계산
    source_crop_x_start = x_starts[source_h_idx]
    source_crop_y_start = y_starts[source_v_idx]
    global_x = source_crop_x_start + local_x
    global_y = source_crop_y_start + local_y
    
    source_img_with_box = img_local_base.copy()
    draw = ImageDraw.Draw(source_img_with_box)
    draw.rectangle(
        [global_x, global_y, global_x + patch_size, global_y + patch_size],
        outline='red', width=2
    )
    
    return source_img_with_box, target_img, similarity_map_img, overlay_img, active_locs
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# --- 4. 메인 실행 부분 --- (이 부분은 변경 없음)
if __name__ == "__main__":
    if not Path(IMAGE_PATH).exists():
        print(f"!! ERROR: Image file not found at '{IMAGE_PATH}'")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        output_dir = Path("visualize/dinov3_scale_final")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Demo a: Global 이미지와 비교 ---
        print("\n--- Running Demo (a): Comparing with Global Image ---")
        SOURCE_LOC_A = 'middle_left'
        K_INDEX_A = 434 
        TARGET_LOC_A = 'global'
        
        results_a = visualize_embedding_similarity(
            device=device, img_path=IMAGE_PATH, vfm_embedding_dir=VFM_EMBEDDING_DIR,
            source_crop_loc=SOURCE_LOC_A, k_index=K_INDEX_A, comparison_target_loc=TARGET_LOC_A
        )

        if results_a:
            source_patch_img_a, target_img_a, sim_map_a, overlay_a, active_locs_a = results_a
            print(f"Image processed. Available local positions: {active_locs_a}")
            file_stem = Path(IMAGE_PATH).stem
            prefix_a = f"{file_stem}_source_{SOURCE_LOC_A}_{K_INDEX_A}_target_{TARGET_LOC_A}"
            source_patch_img_a.save(output_dir / f"{prefix_a}_1_source_patch_location.png")
            target_img_a.save(output_dir / f"{prefix_a}_2_target_image.png")
            sim_map_a.save(output_dir / f"{prefix_a}_3_similarity_map.png")
            overlay_a.save(output_dir / f"{prefix_a}_4_overlay.png")
            print(f"Successfully saved 4 images for Demo (a) to '{output_dir}'")
        else:
            print("Could not generate visualization for Demo (a) due to an error.")

        # --- Demo b: Tile Level 이미지와 비교 ---
        print("\n--- Running Demo (b): Comparing with Another Tile ---")
        SOURCE_LOC_B = 'top_left'
        K_INDEX_B = 189
        TARGET_LOC_B = 'middle_right'

        results_b = visualize_embedding_similarity(
            device=device, img_path=IMAGE_PATH, vfm_embedding_dir=VFM_EMBEDDING_DIR,
            source_crop_loc=SOURCE_LOC_B, k_index=K_INDEX_B, comparison_target_loc=TARGET_LOC_B
        )

        if results_b:
            source_patch_img_b, target_img_b, sim_map_b, overlay_b, _ = results_b
            file_stem = Path(IMAGE_PATH).stem
            prefix_b = f"{file_stem}_source_{SOURCE_LOC_B}_{K_INDEX_B}_target_{TARGET_LOC_B}"
            source_patch_img_b.save(output_dir / f"{prefix_b}_1_source_patch_location.png")
            target_img_b.save(output_dir / f"{prefix_b}_2_target_image.png")
            sim_map_b.save(output_dir / f"{prefix_b}_3_similarity_map.png")
            overlay_b.save(output_dir / f"{prefix_b}_4_overlay.png")
            print(f"Successfully saved 4 images for Demo (b) to '{output_dir}'")
        else:
            print("Could not generate visualization for Demo (b) due to an error.")