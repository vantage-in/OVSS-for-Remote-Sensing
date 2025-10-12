import argparse
import os
import glob
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import v2

def make_transform(resize_size: int = 224):
    """제공된 전처리 파이프라인을 생성합니다."""
    # torchvision.transforms.v2 API를 사용합니다.
    return v2.Compose([
        v2.ToImage(),  # PIL Image를 Tensor로 변환
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True), # 0-255 uint8 -> 0-1 float32
        v2.Normalize(
            mean=[0.430, 0.411, 0.296],
            std=[0.213, 0.156, 0.143],
        ),
    ])

def parse_args():
    """스크립트 실행을 위한 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Extract DINOv3 features for a dataset')
    parser.add_argument('--image-dir', required=True, help='Path to the input directory of images.')
    parser.add_argument('--output-dir', required=True, help='Path to the output directory to save features.')
    # parser.add_argument('--weights-path', required=True, help='Path to the local DINOv3 weights file (e.g., dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth).')
    parser.add_argument('--dinov3-repo-path', default='./dinov3', help='Path to the local dinov3 repository.')
    parser.add_argument('--device', default='cuda:0', help='Device to run the model on (e.g., "cuda:0" or "cpu").')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. 모델 로드
    print(f"Loading DINOv3 model from local repository at '{args.dinov3_repo_path}'...")
    try:
        model = torch.hub.load(
            repo_or_dir=args.dinov3_repo_path,
            model='dinov3_vitl16',
            source='local',
            weights='dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
        )
    except Exception as e:
        print(f"Error loading model. Make sure '{args.dinov3_repo_path}' is a valid DINOv3 repository path.")
        print(f"Details: {e}")
        return

    model.to(device)
    model.eval()
    print("Model loaded successfully and moved to device:", device)

    # 2. 전처리 파이프라인 생성
    transform = make_transform()

    # 3. 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. 이미지 파일 목록 탐색
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg'))) + \
                  sorted(glob.glob(os.path.join(args.image_dir, '*.png'))) + \
                  sorted(glob.glob(os.path.join(args.image_dir, '*.JPG')))

    print(f"Found {len(image_paths)} images to process.")

    # 5. 각 이미지에 대한 피처 추출 및 저장
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(image_paths, desc="Extracting features")):
            try:
                # 이미지 로드 및 전처리
                image = Image.open(img_path).convert('RGB')
                preprocessed_image = transform(image).unsqueeze(0).to(device) # 배치 차원 추가 및 장치로 이동

                # 피처 추출 (일반적으로 CLS 토큰의 임베딩이 반환됨)
                features = model(preprocessed_image)

                # --- [검증용 코드] ---
                # 첫 번째 이미지에 대해서만 피처의 차원을 출력합니다.
                if i == 0:
                    print("\n" + "="*50)
                    print("Feature Dimension Verification (for first image):")
                    print(f"  - Input image tensor shape: {preprocessed_image.shape}")
                    print(f"  - Output feature tensor shape: {features.shape}")
                    print(f"  - The feature dimension is: {features.shape[-1]}")
                    print("="*50 + "\n")
                    # ViT-L 모델의 경우, [1, 1024] 형태가 예상됩니다.

                # 피처 저장
                basename = os.path.basename(img_path)
                filename_without_ext = os.path.splitext(basename)[0]
                output_path = os.path.join(args.output_dir, f"{filename_without_ext}.pt")
                
                # GPU 텐서를 CPU로 옮긴 후 저장
                torch.save(features.squeeze(0).cpu(), output_path)

            except Exception as e:
                print(f"\nCould not process {img_path}. Error: {e}")

    print("\n🎉 Feature extraction complete!")
    print(f"Saved features to: {args.output_dir}")


if __name__ == '__main__':
    main()