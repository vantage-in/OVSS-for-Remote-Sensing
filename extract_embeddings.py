'''
python extract_embeddings.py configs/cfg_vdd.py --work-dir ./work_tmp --output-dir ./embeddings/vdd
'''

import argparse
import os
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from tqdm import tqdm
import os.path as osp

import torch.nn.functional as F


import custom_datasets

def parse_args():
    """스크립트 실행을 위한 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Extract and save embeddings for SegEarth-OV')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('--work-dir', help='the directory to save logs and models')
    parser.add_argument(
        '--output-dir', 
        default='./embeddings', 
        help='directory to save extracted embeddings'
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    # 설정 파일 로드
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
        
    # 출력 디렉토리 생성
    output_dir = args.output_dir
    if cfg.launcher == 'none' or int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(output_dir, exist_ok=True)

    # MMSegmentation Runner 빌드
    runner = Runner.from_cfg(cfg)
    model = runner.model
    data_loader = runner.test_dataloader

    model.eval()
    
    # tqdm을 사용하여 진행 상황 표시
    # 분산 환경에서는 메인 프로세스(rank 0)에서만 tqdm을 활성화합니다.
    is_main_process = cfg.launcher == 'none' or int(os.environ.get('RANK', 0)) == 0
    if is_main_process:
        data_loader = tqdm(data_loader)

    with torch.no_grad():
        for data in data_loader:
            processed_data = model.data_preprocessor(data, training=False)

            img = processed_data['inputs'][0].to('cuda').unsqueeze(0).half()
            sample_object = data['data_samples'][0]
            
            # MMDistributedDataParallel 래퍼를 고려하여 실제 모델에 접근
            segmentor = model.module if hasattr(model, 'module') else model

            #--- 임베딩 추출 ---#
            # ========================================================================
            # Phase 0: Extract global features if needed
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

            global_dino_feats_flat, global_clip_feats_flat = None, None
            if use_global:
                if printing: print("Phase 0: Extracting global context features...")
                with torch.no_grad():
                    # 1. Get original image dimensions
                    h_img, w_img = img.shape[-2:]
                    target_size = 224

                    # 2. Calculate new size to maintain aspect ratio, fitting the longest side to 224
                    scale = target_size / max(h_img, w_img)
                    new_h, new_w = int(h_img * scale), int(w_img * scale)

                    # 3. Resize the image with the correct aspect ratio
                    resized_img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

                    # 4. Calculate padding needed to make the image 224x224
                    pad_h = target_size - new_h
                    pad_w = target_size - new_w
                    pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
                    pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
                    
                    # 5. Add symmetrical padding (left/right, top/bottom)
                    global_view_img = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom))

                    # global_view_img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                    h_gl_tok, w_gl_tok = global_view_img.shape[-2] // segmentor.patch_size[0], global_view_img.shape[-1] // segmentor.patch_size[1]

                    x_global = segmentor.net.encode_before_last_layer(global_view_img)
                    global_dino_feats = segmentor.ref_feature_dino(global_view_img)
                    global_clip_feats = segmentor.net.encode_value_projection(x_global, h_gl_tok, w_gl_tok, target_size=(global_dino_feats.shape[-2:]))

                    # dino_patch_size = segmentor.dino_patch_size
                    # pad_top_feat = pad_top // dino_patch_size
                    # pad_bottom_feat = pad_bottom // dino_patch_size
                    # pad_left_feat = pad_left // dino_patch_size
                    # pad_right_feat = pad_right // dino_patch_size

                    # feat_h, feat_w = global_dino_feats.shape[-2:]

                    # # unpad
                    # global_dino_feats = global_dino_feats[:, :, pad_top_feat : feat_h - pad_bottom_feat, pad_left_feat : feat_w - pad_right_feat]
                    # global_clip_feats = global_clip_feats[:, :, pad_top_feat : feat_h - pad_bottom_feat, pad_left_feat : feat_w - pad_right_feat]
            
                    global_dino_feats_flat = global_dino_feats.flatten(2, 3).permute(0, 2, 1).reshape(-1, global_dino_feats.shape[1])
                    global_clip_feats_flat = global_clip_feats.flatten(2, 3).permute(2, 0, 1).reshape(-1, global_clip_feats.shape[0] * global_clip_feats.shape[1])
                    
                    global_dino_feats_flat = global_dino_feats_flat.to(dtype=torch.float32)
                    global_clip_feats_flat = global_clip_feats_flat.to(dtype=torch.float32)
                    if printing: print(f"  - Extracted {global_dino_feats_flat.shape[0]} global feature pairs.")

            # ========================================================================
            # Phase 1: Collect and sample patch features if needed
            # ========================================================================
            all_last_feats = []

            all_robust_dino_feats, all_robust_clip_feats, all_robust_patch_ids = None, None, None

            if use_sampling:
                if printing: print("Phase 1: Collecting robust feature embeddings and their patch IDs...") 

                patch_counter = 0
                all_robust_dino, all_robust_clip, all_robust_patch_ids = [], [], []

                for h_idx in range(h_grids):
                    for w_idx in range(w_grids):
                        # ... (crop_img 생성 및 강건성 마스크 계산 로직은 동일) ...
                        y1, x1 = h_idx * h_stride, w_idx * w_stride
                        y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
                        y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                        crop_img = img[:, :, y1:y2, x1:x2]

                        H_orig, W_orig = crop_img.shape[-2:]
                        pad = segmentor.compute_padsize(H_orig, W_orig, segmentor.patch_size[0])

                        if any(pad):
                            padded_img = F.pad(crop_img, pad, mode='constant', value=0)
                        else:
                            padded_img = crop_img

                        x = segmentor.net.encode_before_last_layer(padded_img)
                        all_last_feats.append(x)

                        h_pad_tok, w_pad_tok = padded_img.shape[-2] // segmentor.patch_size[0], padded_img.shape[-1] // segmentor.patch_size[1]

                        dino_feat = segmentor.ref_feature_dino(padded_img) 
                        target_h, target_w = dino_feat.shape[-2:]
                        clip_feat = segmentor.net.encode_value_projection(x, h_pad_tok, w_pad_tok, target_size=(target_h, target_w))

                        dino_feat_flat = dino_feat.flatten(2, 3).permute(0, 2, 1).reshape(-1, dino_feat.shape[1])
                        clip_feat_flat = clip_feat.flatten(2, 3).permute(2, 0, 1).reshape(-1, clip_feat.shape[0] * clip_feat.shape[1])

                        all_robust_dino.append(dino_feat_flat)
                        all_robust_clip.append(clip_feat_flat)

                        # [추가] 이 강건한 피처들이 현재 패치(patch_counter) 소속임을 기록
                        num_robust_in_patch = dino_feat_flat.shape[0]
                        all_robust_patch_ids.append(torch.full((num_robust_in_patch,), patch_counter, device=device))

                        patch_counter += 1
            
                # 수집된 모든 강건한 임베딩과 ID를 하나의 텐서로 통합
                if all_robust_dino:
                    all_robust_dino_feats = torch.cat(all_robust_dino, dim=0).to(dtype=torch.float32)
                    all_robust_clip_feats = torch.cat(all_robust_clip, dim=0).to(dtype=torch.float32)
                    all_robust_patch_ids = torch.cat(all_robust_patch_ids, dim=0)

                if printing: 
                    num_total_robust = all_robust_dino_feats.shape[0]
                    print(f"  - Collected {num_total_robust} robust feature embeddings in total.")

            #--- 추출된 텐서를 CPU로 이동시켜 저장 준비 ---#
            embeddings = {
                # 'global_dino_feats_flat': global_dino_feats_flat.cpu(),
                # 'global_clip_feats_flat': global_clip_feats_flat.cpu(),
                # 'all_robust_dino_feats': all_robust_dino_feats.cpu(),
                # 'all_robust_clip_feats': all_robust_clip_feats.cpu(),
                # 'all_robust_patch_ids': all_robust_patch_ids.cpu(),
                # 'all_last_feats': [t.cpu() for t in all_last_feats],
                '__debug_inputs': img.cpu()
            }

            #--- 파일로 저장 ---#
            # 각 이미지의 고유한 파일명을 사용하여 저장합니다.
            img_path = sample_object.img_path
            basename = osp.basename(img_path)
            filename_without_ext = osp.splitext(basename)[0]
            output_path = osp.join(output_dir, f"{filename_without_ext}.pt")
            torch.save(embeddings, output_path)

    if is_main_process:
        print(f"Embeddings extracted and saved to {output_dir}")

if __name__ == '__main__':
    main()