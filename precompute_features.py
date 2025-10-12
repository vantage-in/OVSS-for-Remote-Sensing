'''
python precompute_features.py \
    configs/cfg_openearthmap.py \
    path/to/your/checkpoint.pth \
    --out precomputed_features/openearthmap
'''
import argparse
import os
import os.path as osp

import mmengine
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from proxy_segearth_segmentor_cat_random import ProxySegEarthSegmentationCatRandom
from open_clip.model import CLIP
from open_clip.transformer import VisionTransformer

# ------------------------------------------------------------------------------
# 사전 계산을 위한 새로운 모델 메서드
# ------------------------------------------------------------------------------
@torch.no_grad()
def extract_all_features(model: ProxySegEarthSegmentationCatRandom, img: torch.Tensor, img_metas: dict):
    """
    forward_slide의 Phase 0과 1 로직을 수행하여
    모든 중간 특징들을 추출하고 딕셔너리로 반환합니다.
    """
    model.eval()
    
    # 기본 파라미터 설정
    stride, crop_size = model.slide_stride, model.slide_crop
    h_stride, w_stride = (stride, stride) if isinstance(stride, int) else stride
    h_crop, w_crop = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    
    # 비정방형 이미지를 위한 로직 (H, W 통일)
    img_h, img_w = img.shape[-2:]
    device = img.device

    # ========================================================================
    # Phase 0: Global 특징 추출
    # ========================================================================
    global_dino_feats_flat, global_clip_feats_flat = None, None
    
    # 비율 유지 리사이즈 및 패딩
    target_size = 224
    scale = target_size / max(img_h, img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    resized_img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
    pad_h, pad_w = target_size - new_h, target_size - new_w
    pad_top, pad_bottom = pad_h // 2, pad_h - (pad_h // 2)
    pad_left, pad_right = pad_w // 2, pad_w - (pad_w // 2)
    global_view_img = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom))

    patch_size = 16
    h_gl_tok, w_gl_tok = global_view_img.shape[-2] // patch_size, global_view_img.shape[-1] // patch_size
    
    # 특징 추출
    x_global = model.net.encode_before_last_layer(global_view_img)
    global_dino_feats = model.ref_feature_dino(global_view_img)
    
    global_clip_feats = model.net.encode_value_projection(x_global, h_gl_tok, w_gl_tok, target_size=(global_dino_feats.shape[-2:]))

    global_dino_feats_flat = global_dino_feats.flatten(2, 3).permute(0, 2, 1).reshape(-1, global_dino_feats.shape[1])
    global_clip_feats_flat = global_clip_feats.flatten(2, 3).permute(2, 0, 1).reshape(-1, global_clip_feats.shape[0] * global_clip_feats.shape[1])

    # ========================================================================
    # Phase 1: Patch 특징 수집
    # ========================================================================
    h_grids = max(img_h - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(img_w - w_crop + w_stride - 1, 0) // w_stride + 1
    
    all_robust_dino, all_robust_clip, all_robust_patch_ids_list = [], [], []
    all_last_feats = []
    patch_counter = 0

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1, x1 = h_idx * h_stride, w_idx * w_stride
            y2, x2 = min(y1 + h_crop, img_h), min(x1 + w_crop, img_w)
            y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]

            h_crop_orig, w_crop_orig = crop_img.shape[-2:]
            pad = model.compute_padsize(h_crop_orig, w_crop_orig, model.patch_size[0])
            padded_img = F.pad(crop_img, pad, mode='constant', value=0) if any(pad) else crop_img
            
            h_pad_tok, w_pad_tok = padded_img.shape[-2] // patch_size, padded_img.shape[-1] // patch_size

            x = model.net.encode_before_last_layer(padded_img)
            all_last_feats.append(x.cpu()) # CPU로 이동하여 저장

            dino_feat = model.ref_feature_dino(padded_img)
            
            # 비정방형 처리를 위해 동적 target_size 전달
            target_h_dino, target_w_dino = dino_feat.shape[-2:]
            clip_feat = model.net.encode_value_projection(x, h_pad_tok, w_pad_tok, target_size=(target_h_dino, target_w_dino))

            dino_feat_flat = dino_feat.flatten(2, 3).permute(0, 2, 1).reshape(-1, dino_feat.shape[1])
            clip_feat_flat = clip_feat.flatten(2, 3).permute(2, 0, 1).reshape(-1, clip_feat.shape[0] * clip_feat.shape[1])

            all_robust_dino.append(dino_feat_flat)
            all_robust_clip.append(clip_feat_flat)
            
            num_robust_in_patch = dino_feat_flat.shape[0]
            all_robust_patch_ids_list.append(torch.full((num_robust_in_patch,), patch_counter, device=device))
            patch_counter += 1

    all_robust_dino_feats = torch.cat(all_robust_dino, dim=0) if all_robust_dino else torch.empty(0)
    all_robust_clip_feats = torch.cat(all_robust_clip, dim=0) if all_robust_clip else torch.empty(0)
    all_robust_patch_ids = torch.cat(all_robust_patch_ids_list, dim=0) if all_robust_patch_ids_list else torch.empty(0)

    # 최종적으로 저장할 딕셔너리 구성 (CPU 텐서로)
    features_to_save = {
        'global_dino': global_dino_feats_flat.cpu(),
        'global_clip': global_clip_feats_flat.cpu(),
        'patch_dino': all_robust_dino_feats.cpu(),
        'patch_clip': all_robust_clip_feats.cpu(),
        'patch_ids': all_robust_patch_ids.cpu(),
        'last_feats': all_last_feats # 이미 CPU 텐서 리스트
    }
    
    return features_to_save

# ------------------------------------------------------------------------------
# 메인 스크립트 로직
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Pre-compute features for ablation studies')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output directory to save feature files')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # 1. 설정 파일 로드
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs', 'precompute_tmp')

    # 2. 모델 및 데이터 로더 빌드
    runner = Runner.from_cfg(cfg)
    model = runner.model
    # 체크포인트 로드
    if args.checkpoint is not None:
        from mmengine.runner.checkpoint import load_checkpoint
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    # GPU로 모델 이동
    if torch.cuda.is_available():
        model = model.cuda()
    
    dataloader = runner.test_dataloader

    # 3. 저장 디렉토리 설정
    if args.out:
        out_dir = args.out
    else:
        # 데이터셋 이름으로 자동 디렉토리 생성
        dataset_name = dataloader.dataset.__class__.__name__
        out_dir = osp.join('./precomputed_features', dataset_name)
    
    mmengine.mkdir_or_exist(out_dir)
    print(f"Features will be saved to: {out_dir}")

    # 4. 데이터셋 순회 및 특징 추출/저장
    progress_bar = mmengine.ProgressBar(len(dataloader.dataset))
    for data in dataloader:
        if torch.cuda.is_available():
            data = runner.collate_fn(data)
            data['inputs'][0] = data['inputs'][0].cuda()

        img = data['inputs'][0]
        img_metas = data['data_samples'][0].metainfo
        
        # 특징 추출
        features = extract_all_features(model, img, img_metas)
        
        # 파일 경로 생성 및 저장
        img_filename = osp.basename(img_metas['img_path'])
        save_name = f"{osp.splitext(img_filename)[0]}.pt"
        save_path = osp.join(out_dir, save_name)
        
        torch.save(features, save_path)
        progress_bar.update()

    print("\nFeature pre-computation completed.")


if __name__ == '__main__':
    main()