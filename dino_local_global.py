import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2
from torchvision import transforms
from pathlib import Path # 파일 경로 처리를 위해 추가

# --- 1. 설정 변수 ---
# 사용자가 변경할 수 있는 파라미터들
IMAGE_PATH = 'demo/image/kyoto_33.tif' # ★ 1. 로컬 이미지 파일 경로로 변경
CROP_LOCATION = 'bottom_left'  # 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'center' 중 선택
K_PATCH_INDEX = 320      # 448x488 이미지에서 크롭한 영역의 k번째 패치 (0 ~ 783 사이의 정수)

# --- 2. 모델 및 전처리기 준비 ---
def setup_model_and_transforms():
    """DINO 모델과 이미지 전처리기를 로드하고 설정합니다."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # DINO ViT-b/8 모델 로드
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').to(device)
    model.eval()

    # 이미지 전처리기
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    
    return model, transform, device

def get_dino_patch_embeddings(image_tensor, model, device):
    """주어진 이미지 텐서에서 DINO 패치 임베딩을 추출합니다."""
    with torch.no_grad():
        embeddings = model.get_intermediate_layers(image_tensor.to(device))[0]
        patch_embeddings = embeddings[0, 1:, :]
    return patch_embeddings

def get_crop_box(size=(448, 448), crop_size=(224, 224), location='center'):
    """지정된 위치에 대한 crop box 좌표를 반환합니다."""
    width, height = size
    crop_width, crop_height = crop_size
    
    if location == 'top_left':
        return (0, 0, crop_width, crop_height)
    elif location == 'top_right':
        return (width - crop_width, 0, width, crop_height)
    elif location == 'bottom_left':
        return (0, height - crop_height, crop_width, height)
    elif location == 'bottom_right':
        return (width - crop_width, height - crop_height, width, height)
    elif location == 'center':
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return (left, top, left + crop_width, top + crop_height)
    else:
        raise ValueError("Invalid crop location")

# --- 3. 핵심 시각화 함수 ---
def visualize_embedding_similarity(model, transform, device, img_path, crop_loc, k_index):
    """
    스케일이 다른 이미지 간의 DINO 임베딩 유사도를 계산하고 시각화합니다.
    """
    try:
        # ★ 로컬 파일 경로에서 직접 이미지를 열도록 수정
        original_img = Image.open(img_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: The file was not found at {img_path}")
        return None, None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    # --- Step 1: 전체 이미지를 224x224로 리사이즈 후 DINO 임베딩 추출 ---
    img_224 = original_img.resize((224, 224))#, Image.LANCZOS)
    img_tensor_224 = transform(img_224).unsqueeze(0)
    embeddings_224 = get_dino_patch_embeddings(img_tensor_224, model, device)

    # --- Step 2: 전체 이미지를 448x448로 리사이즈 후 특정 224x224 영역을 crop하여 임베딩 추출 ---
    img_448 = original_img.resize((448, 448))#, Image.LANCZOS)
    crop_box = get_crop_box(size=(448, 448), crop_size=(224, 224), location=crop_loc)
    img_448_cropped = img_448.crop(crop_box)
    
    img_tensor_448_cropped = transform(img_448_cropped).unsqueeze(0)
    embeddings_448_cropped = get_dino_patch_embeddings(img_tensor_448_cropped, model, device)

    # --- Step 4: k번째 임베딩과 224x224 전체 임베딩 간의 코사인 유사도 맵 생성 ---
    if not (0 <= k_index < embeddings_448_cropped.shape[0]):
        raise ValueError(f"k_patch_index must be between 0 and {embeddings_448_cropped.shape[0]-1}")

    target_embedding = embeddings_448_cropped[k_index].unsqueeze(0)
    cosine_sim = F.cosine_similarity(target_embedding, embeddings_224, dim=1)
    sim_map = cosine_sim.reshape(28, 28).cpu().numpy()

    # --- Step 5: 유사도 맵을 224x224로 보간 후 원본 이미지에 오버레이 ---
    sim_map_resized = cv2.resize(sim_map, (224, 224), interpolation=cv2.INTER_LINEAR)
    sim_map_clipped = np.clip(sim_map_resized, 0, 0.5)
    sim_map_normalized = sim_map_clipped / 0.5
    sim_map_uint8 = np.uint8(255 * sim_map_normalized)
    heatmap = cv2.applyColorMap(sim_map_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_img = Image.blend(img_224, Image.fromarray(heatmap), alpha=0.5)

    # --- Step 6: 원본 448x448 이미지 상에서 선택한 패치 위치 표시 ---
    patch_size = 8
    grid_size = 28
    patch_row = k_index // grid_size
    patch_col = k_index % grid_size
    local_x = patch_col * patch_size
    local_y = patch_row * patch_size
    global_x = crop_box[0] + local_x
    global_y = crop_box[1] + local_y
    
    img_448_with_box = img_448.copy()
    draw = ImageDraw.Draw(img_448_with_box)
    draw.rectangle(
        [global_x, global_y, global_x + patch_size, global_y + patch_size],
        outline='red', width=2
    )
    
    return overlay_img, img_448_with_box

# --- 4. 메인 실행 부분 ---
if __name__ == "__main__":
    # 모델 및 전처리기 로드
    dino_model, dino_transform, dino_device = setup_model_and_transforms()

    # 시각화 생성
    print(f"Processing image: {IMAGE_PATH}")
    similarity_overlay, marked_original = visualize_embedding_similarity(
        model=dino_model,
        transform=dino_transform,
        device=dino_device,
        img_path=IMAGE_PATH,
        crop_loc=CROP_LOCATION,
        k_index=K_PATCH_INDEX
    )

    # 시각화 결과가 성공적으로 생성되었을 경우에만 저장 진행
    if similarity_overlay is not None and marked_original is not None:
        print("Visualization generation complete.")

        # --- ★ Step 7: 두 이미지를 콜라주하여 하나로 저장 ---
        similarity_overlay_resized = similarity_overlay.resize((448, 448), Image.LANCZOS)
        
        collage = Image.new('RGB', (448 * 2, 448))
        collage.paste(marked_original, (0, 0))
        collage.paste(similarity_overlay_resized, (448, 0))

        # ★ 2, 3, 4. 파일 이름 동적 생성
        # 원본 파일 경로에서 확장자를 제외한 파일 이름 추출
        file_stem = Path(IMAGE_PATH).stem
        # 최종 파일명 조합
        output_filename = f"visualize/dino_scale/dino_scale_{file_stem}_{CROP_LOCATION}_{K_PATCH_INDEX}.png"

        # 결과 저장
        collage.save(output_filename)
        print(f"Collage image saved as '{output_filename}'")
        # collage.show()
    else:
        print("Could not generate visualization due to an error.")