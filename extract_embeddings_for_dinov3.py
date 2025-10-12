"""
python extract_embeddings_for_dinov3.py --image_dir data/VDD/test/src --output_dir embeddings/dinov3/vdd
"""

import torch
import numpy as np
import cv2
import torchvision.transforms as T 
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
from pathlib import Path

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract DINOv3 embeddings from a directory of images.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output .pt files.')
    return parser.parse_args()

def preprocess_image(image_path, target_size=448):
    """
    Replicates the exact mmsegmentation preprocessing pipeline.
    """
    # 1. Load Image with OpenCV (provides BGR numpy array)
    # This corresponds to the `LoadImageFromFile` transform.
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # 2. Resize with OpenCV, keeping aspect ratio.
    # This corresponds to the `Resize` transform with keep_ratio=True.
    h, w, _ = image_bgr.shape
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # The following steps replicate `SegDataPreProcessor`.
    
    # 3. Channel Conversion: BGR -> RGB
    # This happens first in the SegDataPreProcessor forward pass.
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # 4. Convert to float32
    img_float = resized_rgb.astype(np.float32)
    
    # 5. Normalize the RGB image
    # Note: The mean/std values are applied to the RGB channels respectively.
    mean = np.array([122.771, 116.746, 104.094], dtype=np.float32)
    std = np.array([68.501, 66.632, 70.323], dtype=np.float32)
    normalized_img = (img_float - mean) / std

    # 6. HWC to CHW format for PyTorch
    img_chw = normalized_img.transpose(2, 0, 1)

    # 7. Convert to PyTorch Tensor, add batch dimension, and set to half precision
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(torch.float16)
    
    return img_tensor

def normalize_for_dino(img):
    """
    Un-normalizes a tensor from mmsegmentation format and re-normalizes it
    for DINOv3.

    Args:
        tensor (torch.Tensor): Input tensor normalized with mmsegmentation stats.
                               Shape: (1, 3, H, W), dtype: torch.float16

    Returns:
        torch.Tensor: Tensor re-normalized for DINOv3.
    """
    unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    imgs_norm = [norm(unnorm(img[i])) for i in range(len(img))]
    imgs_norm = torch.stack(imgs_norm, dim=0)

    imgs_norm = imgs_norm.half()
    return imgs_norm

def compute_padsize(H: int, W: int, patch_size: int):
    l, r, t, b = 0, 0, 0, 0
    if W % patch_size:
        lr = patch_size - (W % patch_size)
        l = lr // 2
        r = lr - l

    if H % patch_size:
        tb = patch_size - (H % patch_size)
        t = tb // 2
        b = tb - t

    return l, r, t, b

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"
    DINOV3_LOCATION = "dinov3" # Or DINOV3_GITHUB_LOCATION

    MODEL_NAME = "dinov3_vitl16"

    # model = torch.hub.load(
    #     repo_or_dir=DINOV3_LOCATION,
    #     model=MODEL_NAME,
    #     source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
    # )
    model = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source='local',
        weights='dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    )

    model.cuda().eval()

    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = sorted([p for p in Path(args.image_dir).glob('*') if p.suffix.lower() in image_extensions])
    
    if not image_paths:
        print(f"경고: '{args.image_dir}'에서 처리할 이미지를 찾지 못했습니다.")
        return

    print(f"Found {len(image_paths)} images in '{args.image_dir}'. Processing...")


    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Extracting Embeddings"):
            
            img = preprocess_image(str(image_path))
            img = normalize_for_dino(img).cuda()
            
            # ========================================================================
            # Phase 0: Extract global features 
            # ========================================================================
            printing = False
            use_global = True
            use_sampling = True

            stride = (112, 112)
            crop_size = (224, 224)
            h_stride, w_stride = stride
            h_crop, w_crop = crop_size
            batch_size, _, h_img, w_img = img.shape
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            device = img.device

            global_dino_feats_flat = None
            if use_global:
                if printing: print("Phase 0: Extracting global context features...")
                with torch.no_grad():
                    # 1. Get original image dimensions
                    h_img, w_img = img.shape[-2:]
                    target_size = 448

                    # 4. Calculate padding needed to make the image 224x224
                    pad_h = target_size - new_h
                    pad_w = target_size - new_w
                    pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
                    pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
                    
                    # 5. Add symmetrical padding (left/right, top/bottom)
                    global_view_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom))

                    # global_view_img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                    h_gl_tok, w_gl_tok = global_view_img.shape[-2] // 16, global_view_img.shape[-1] // 16

                    global_dino_feats = model.get_intermediate_layers(global_view_img, n=1, reshape=True, return_class_token=False, norm=True)[0] # [B, C, H, W]
                    assert(global_dino_feats[2] == 28 and global_dino_feats[3] == 28, "wrong shape")

                    global_dino_feats_flat = global_dino_feats.flatten(2, 3).permute(0, 2, 1).reshape(-1, global_dino_feats.shape[1])
   
                    global_dino_feats_flat = global_dino_feats_flat.to(dtype=torch.float32)
                    if printing: print(f"  - Extracted {global_dino_feats_flat.shape[0]} global feature pairs.")

            # ========================================================================
            # Phase 1: Collect and sample patch features if needed
            # ========================================================================
            all_robust_dino_feats = None

            if use_sampling:
                if printing: print("Phase 1: Collecting robust feature embeddings and their patch IDs...") 

                patch_counter = 0
                all_robust_dino, all_robust_patch_ids = [], [], []

                for h_idx in range(h_grids):
                    for w_idx in range(w_grids):
                        # ... (crop_img 생성 및 강건성 마스크 계산 로직은 동일) ...
                        y1, x1 = h_idx * h_stride, w_idx * w_stride
                        y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
                        y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                        crop_img = img[:, :, y1:y2, x1:x2]

                        H_orig, W_orig = crop_img.shape[-2:]
                        pad = compute_padsize(H_orig, W_orig, 16)

                        if any(pad):
                            padded_img = F.pad(crop_img, pad, mode='constant', value=0)
                        else:
                            padded_img = crop_img

                        upsized_img = F.interpolate(padded_img, size=(448, 448), mode='bilinear', align_corners=False)
                        dino_feat = model.get_intermediate_layers(upsized_img, n=1, reshape=True, return_class_token=False, norm=True)[0] # [B, C, H, W]

                        dino_feat_flat = dino_feat.flatten(2, 3).permute(0, 2, 1).reshape(-1, dino_feat.shape[1])

                        all_robust_dino.append(dino_feat_flat)

                        # [추가] 이 강건한 피처들이 현재 패치(patch_counter) 소속임을 기록
                        num_robust_in_patch = dino_feat_flat.shape[0]
                        all_robust_patch_ids.append(torch.full((num_robust_in_patch,), patch_counter, device=device))

                        patch_counter += 1
            
                # 수집된 모든 강건한 임베딩과 ID를 하나의 텐서로 통합
                if all_robust_dino:
                    all_robust_dino_feats = torch.cat(all_robust_dino, dim=0).to(dtype=torch.float32)
                    all_robust_patch_ids = torch.cat(all_robust_patch_ids, dim=0)

                if printing: 
                    num_total_robust = all_robust_dino_feats.shape[0]
                    print(f"  - Collected {num_total_robust} robust feature embeddings in total.")

            #--- 추출된 텐서를 CPU로 이동시켜 저장 준비 ---#
            embeddings = {
                'global_dino_feats_flat': global_dino_feats_flat.cpu(),
                'all_robust_dino_feats': all_robust_dino_feats.cpu(),
                # 'all_robust_patch_ids': all_robust_patch_ids.cpu(),
                # '__debug_inputs': img.cpu()
            }

            # 각 이미지의 고유한 파일명을 사용하여 저장합니다.
            output_filename = Path(args.output_dir) / f"{image_path.stem}.pt"
            torch.save(embeddings, output_filename)
            
    print(f"Embeddings extracted and saved to {output_dir}")

if __name__ == '__main__':
    main()