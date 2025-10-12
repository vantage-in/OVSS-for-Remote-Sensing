import torch
import torch.nn as nn
import sys

sys.path.append("..")

from prompts.imagenet_template import *

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

import torch.nn.functional as F

from open_clip import tokenizer, create_model
from BLIP.models.blip_retrieval import blip_retrieval
import gem
from simfeatup_dev.upsamplers import get_upsampler

import torchvision.transforms as T  
from typing import Optional
import math, os
import os.path as osp
import numpy as np

from myutils import UnNormalize
from segment_anything import sam_model_registry
# from sklearn.cluster import KMeans
from fast_pytorch_kmeans import KMeans
from kornia.filters import gaussian_blur2d
import torchvision.transforms.functional as TF

import cv2

@MODELS.register_module()
class ProxySegEarthSegmentationCatRandom(BaseSegmentor):
    def __init__(self,
                 clip_type,
                 vit_type,
                 model_type,
                 name_path,
                 device=torch.device('cuda'),
                 ignore_residual=True,
                 prob_thd=0.0,
                 logit_scale=50,
                 slide_stride=112,
                 slide_crop=224,
                 cls_token_lambda=0,
                 bg_idx=0,
                 feature_up=True,
                 feature_up_cfg=dict(
                     model_name='jbu_one',
                     model_path='your/model/path'),
                 cls_variant: Optional[str] = None,
                 embedding_dir: Optional[str] = None,
                 context_mode: Optional[str] = None,
                 vfm_model=None,
                 **kwargs):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True)
        super().__init__(data_preprocessor=data_preprocessor)
        if clip_type == 'CLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='openai', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='openai', precision='fp16')
        elif clip_type == 'RemoteCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RemoteCLIP-ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RemoteCLIP-ViT-L-14.pt', precision='fp16')
        elif clip_type == 'GeoRSCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RS5M_ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RS5M_ViT-L-14.pt', precision='fp16')
            elif 'H' in vit_type:
                self.net = create_model('ViT-H-14', pretrained='checkpoint/RS5M_ViT-H-14.pt', precision='fp16')
        elif clip_type == 'SkyCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', \
                                        pretrained='checkpoint/SkyCLIP_ViT_B32_top50pct/epoch_20.pt', \
                                        precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', \
                                        pretrained='checkpoint/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt', \
                                        precision='fp16')
        elif clip_type == 'OpenCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='laion2b_s34b_b88k', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='laion2b_s32b_b82k', precision='fp16')
        elif clip_type == 'MetaCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B-16-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L/14-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
        elif clip_type == 'BLIP':
            if 'B' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_base_14M.pth', image_size=slide_crop, vit='base')
            elif 'L' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_large.pth', image_size=slide_crop, vit='large')
            self.net = self.net.half()
        elif clip_type == 'ALIP':
            self.net = create_model('ViT-B/32', pretrained='checkpoint/ALIP_YFCC15M_B32.pt', precision='fp16')

        if model_type == 'GEM':
            if 'B' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'laion2b_s34b_b88k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-B/16-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            elif 'L' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'laion2b_s32b_b82k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-L-14-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            self.net = self.net.model

        self.net.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.clip_type = clip_type
        self.vit_type = vit_type
        self.model_type = model_type
        self.feature_up = feature_up
        self.cls_token_lambda = cls_token_lambda
        self.output_cls_token = cls_token_lambda != 0
        self.bg_idx = bg_idx

        if self.clip_type == 'BLIP':
            self.patch_size = self.net.visual_encoder.patch_size
        else:
            self.patch_size = self.net.visual.patch_size

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad(): # sub_imagenet_template, openai_imagenet_template
            for qw in query_words:
                if self.clip_type == 'BLIP':
                    query =self.net.tokenizer([temp(qw) for temp in openai_imagenet_template], padding='max_length',
                                           truncation=True, max_length=35,
                                           return_tensors="pt").to(device)
                    text_output = self.net.text_encoder(query.input_ids, attention_mask=query.attention_mask,
                                                        mode='text')
                    feature = F.normalize(self.net.text_proj(text_output.last_hidden_state[:, 0, :]))
                else:
                    query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.ignore_residual = ignore_residual
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        if feature_up:
            self.feat_dim = self.query_features.shape[-1]
            self.upsampler = get_upsampler(feature_up_cfg['model_name'], self.feat_dim).cuda().half()
            ckpt = torch.load(feature_up_cfg['model_path'])['state_dict']
            weights_dict = {k[10:]: v for k, v in ckpt.items()}
            self.upsampler.load_state_dict(weights_dict, strict=True)

        self.cls_variant = 'none' if cls_variant is None else cls_variant.lower()

        self.vfm_model = vfm_model
        if vfm_model == 'sam':
            checkpoint = None
            self.vfm = sam_model_registry["vit_b"](checkpoint=checkpoint)
            # self.vfm = sam_model_registry["vit_l"](checkpoint=checkpoint)
        elif vfm_model == 'dino':
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
            self.dino_patch_size = 8
        elif vfm_model == 'dinov2':
            # self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            self.dino_patch_size = 16
        elif vfm_model == 'mae':
            self.vfm = models_vit.__dict__['vit_base_patch16'](img_size=slide_crop, num_classes=0, global_pool=False)
            checkpoint_model = torch.load(checkpoint, map_location='cpu')['model']
            state_dict = self.vfm.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(self.vfm, checkpoint_model)
            # load pre-trained model
            self.vfm.load_state_dict(checkpoint_model, strict=False)
        else:
            print("vlm_model not supported")

        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.num_head = 12
        self.head_dim = 64

        self.embedding_dir = embedding_dir
        self.context_mode = context_mode

    @torch.no_grad()
    def compute_cls_logits(self, img: torch.Tensor,
                        mode: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            img  : [1,3,H,W] 0-1 RGB Tensor
            mode : "blur" | "noise" (None → self.cls_variant)
        Returns:
            cls_logits : [1, n_queries]
        """
        mode = (mode or self.cls_variant).lower()
        if mode not in {"none", "blur", "noise"}:
            raise ValueError("cls_variant must be none | blur | noise")

        # ── 1) 이미지 변형 선택 ───────────────────────
        if mode == "blur":
            sigma = 2.0
            v_img = T.GaussianBlur(kernel_size=int(max(3, 4 * sigma + 1)), sigma=sigma)(img)
        elif mode == "noise":
            v_img = (img + 0.2 * torch.randn_like(img)).clamp(0, 1)
        else:                                          # "none"
            v_img = img                                # 변형하지 않음

        # ── 2) CLIP 인코더 통과 → CLS 토큰 ─────────────
        if self.clip_type == 'BLIP':
            v_resize = F.interpolate(v_img, size=(self.slide_crop, self.slide_crop),
                                    mode='bilinear', align_corners=False)
            cls_emb = self.net.visual_encoder(v_resize, self.ignore_residual)[:, 0, :]
            cls_emb = self.net.vision_proj(cls_emb)         # [1, D]
        else:
            cls_emb, _ = self.net.encode_image(
                v_img.half(), self.model_type, ex_feats=None,
                ignore_residual=self.ignore_residual, output_cls_token=True)  # [1, D]

        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        cls_logits = cls_emb @ self.query_features.T          # [1, Q]
        return cls_logits

    @torch.no_grad()
    def _calculate_hf_score_multiscale_torch(self, patch_tensor: torch.Tensor, sigmas=[(1,2), (1,6), (4,8), (8,16), (16,32), (32,64)]) -> tuple[float, list[float]]:
        """
        [PyTorch Version] NumPy/OpenCV 버전과 100% 동일한 결과를 내면서
        GPU에서 모든 연산을 수행하여 속도를 향상시킨 버전.
        """
        # 1. 텐서 준비 (Un-normalize 및 0-255 uint8 스케일링)
        if patch_tensor.dim() == 4 and patch_tensor.shape[0] == 1:
            patch_tensor = patch_tensor.squeeze(0)
        unnormalized_tensor = self.unnorm(patch_tensor)

        # 2. RGB to Grayscale 변환 (uint8)
        img_uint8 = (unnormalized_tensor * 255).round().clamp(0, 255).to(torch.uint8)
        gray_uint8 = (0.299 * img_uint8[0] + 0.587 * img_uint8[1] + 0.114 * img_uint8[2]).to(torch.uint8)
        
        # 3. e_tot 계산 (uint8 제곱 오버플로우 재현)
        # 제곱 연산 시 uint8 오버플로우가 발생하고, 합산 시 오버플로우를 막기 위해 long으로 캐스팅
        e_tot = torch.sum((gray_uint8**2).long()) + 1e-8
        
        scores = []
        # 4. 각 시그마 스케일에 대해 반복
        for s1, s2 in sigmas:
            # 4-1. PyTorch 기반 GaussianBlur 적용 (uint8 입력 -> uint8 출력)
            blur1 = self._gaussian_blur_opencv_like_torch(gray_uint8, sigma=s1)
            blur2 = self._gaussian_blur_opencv_like_torch(gray_uint8, sigma=s2)
            
            # 4-2. uint8 상태에서 DoG 계산 (wrap-around 뺄셈 재현)
            dog = blur1 - blur2
            
            # 4-3. e_hf 계산 (uint8 제곱 오버플로우 재현)
            e_hf = torch.sum((dog**2).long())
            
            # 4-4. 현재 스케일의 hf_score 계산 및 저장
            scores.append(e_hf / e_tot)
            
        # 5. 가장 큰 점수와 점수 리스트 전체를 반환
        max_score = torch.max(torch.tensor(scores)).item() if scores else 0.0
        
        return max_score, [s.item() for s in scores]

    @torch.no_grad()
    def _calculate_hf_score_torch(self, patch_tensor: torch.Tensor, sigma1: float = 1.0, sigma2: float = 3.0) -> float:
        """
        [PyTorch Version] NumPy/OpenCV 버전과 100% 동일한 결과를 내면서
        GPU에서 모든 연산을 수행하여 속도를 향상시킨 단일 스케일 버전.
        """
        # 1. 텐서 준비 (Un-normalize 및 0-255 uint8 스케일링)
        if patch_tensor.dim() == 4 and patch_tensor.shape[0] == 1:
            patch_tensor = patch_tensor.squeeze(0)
        unnormalized_tensor = self.unnorm(patch_tensor)

        # 2. RGB to Grayscale 변환 (uint8)
        img_uint8 = (unnormalized_tensor * 255).round().clamp(0, 255).to(torch.uint8)
        gray_uint8 = (0.299 * img_uint8[0] + 0.587 * img_uint8[1] + 0.114 * img_uint8[2]).to(torch.uint8)
        
        # 3. PyTorch 기반 GaussianBlur 적용 (uint8 입력 -> uint8 출력)
        blur1 = self._gaussian_blur_opencv_like_torch(gray_uint8, sigma=sigma1)
        blur2 = self._gaussian_blur_opencv_like_torch(gray_uint8, sigma=sigma2)
        
        # 4. uint8 상태에서 DoG 계산 (wrap-around 뺄셈 재현)
        dog = blur1 - blur2
        
        # 5. 에너지 비율 계산 (uint8 제곱 오버플로우 재현)
        e_hf = torch.sum((dog**2).long())
        e_tot = torch.sum((gray_uint8**2).long()) + 1e-8
        hf_score = (e_hf / e_tot).item()

        return hf_score

    def _get_gaussian_kernel_torch(self, ksize: int, sigma: float, device: torch.device) -> torch.Tensor:
        """PyTorch 1D 가우시안 커널을 생성합니다."""
        center = ksize // 2
        x = torch.arange(ksize, dtype=torch.float32, device=device) - center
        kernel1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return kernel1d / kernel1d.sum()

    def _gaussian_blur_opencv_like_torch(self, x_uint8: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        OpenCV와 동일하게 동작하는 PyTorch 기반 가우시안 블러.
        uint8 텐서를 입력받아 uint8 텐서를 반환합니다.
        """
        # 1. OpenCV와 동일한 커널 크기 결정
        ksize = int(round(sigma * 3)) * 2 + 1
        
        # 💡 --- 핵심 수정: 커널 크기 제한 ---
        # 입력 텐서의 높이와 너비 가져오기
        h, w = x_uint8.shape[-2:]
        # ksize가 높이나 너비보다 크지 않도록 제한하고, 홀수로 유지
        ksize = min(ksize, h - 1, w - 1)
        ksize = ksize if ksize % 2 != 0 else ksize - 1
        ksize = max(ksize, 1)
        # --- 여기까지 수정 ---

        # 2. 커널 생성
        kernel1d = self._get_gaussian_kernel_torch(ksize, sigma, x_uint8.device)
        kernel2d = torch.outer(kernel1d, kernel1d).unsqueeze(0).unsqueeze(0) # [1, 1, ksize, ksize]
        
        # 3. uint8 텐서를 float32로 변환하여 컨볼루션 준비
        x_float32 = x_uint8.float().unsqueeze(0).unsqueeze(0)

        # 4. BORDER_REFLECT_101 방식의 패딩 적용
        padding = ksize // 2
        padded_x = F.pad(x_float32, (padding, padding, padding, padding), mode='reflect')
        
        # 5. 2D 컨볼루션으로 블러 적용
        blurred_float = F.conv2d(padded_x, kernel2d, padding='valid').squeeze(0).squeeze(0)
        
        # 6. 다시 uint8 타입으로 변환하여 반환
        return blurred_float.round().clamp(0, 255).to(torch.uint8)

    @torch.no_grad()
    def _calculate_hf_score_multiscale(self, patch_tensor: torch.Tensor, sigmas=[(1,2), (1,6), (4,8), (8,16), (16,32), (32,64)]) -> tuple[float, list[float]]:
        """
        [NEW] 여러 스케일의 DoG를 계산하여 가장 큰 hf_score와 각 스케일별 점수 리스트를 반환합니다.
        사용자의 레퍼런스 코드와 동일하게 uint8 타입으로 DoG를 계산합니다.
        """
        # 1. 텐서를 NumPy 배열로 변환 (기존과 동일)
        if patch_tensor.dim() == 4 and patch_tensor.shape[0] == 1:
            patch_tensor = patch_tensor.squeeze(0)
        patch_tensor = self.unnorm(patch_tensor)

        img_np = patch_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 2. 흑백 변환 (결과는 uint8)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        e_tot = np.sum(gray**2) + 1e-8
        
        scores = []
        # 4. 각 시그마 스케일에 대해 반복
        for s1, s2 in sigmas:
            # 4-1. uint8 이미지에 GaussianBlur 적용
            blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=s1)
            blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=s2)
            
            # 4-2. uint8 상태에서 DoG 계산 (wrap-around 효과 재현)
            dog = blur1 - blur2
            
            # 4-3. e_hf 계산 (float32로 변환 후)
            e_hf = np.sum(dog**2)
            
            # 4-4. 현재 스케일의 hf_score 계산 및 저장
            scores.append(e_hf / e_tot)
            
        # 5. 가장 큰 점수와 점수 리스트 전체를 반환
        max_score = float(np.max(scores)) if scores else 0.0
        
        return max_score, scores

    @torch.no_grad()
    def _calculate_hf_score(self, patch_tensor: torch.Tensor, sigma1: float = 1.0, sigma2: float = 3.0) -> float:
        """
        [수정됨] PyTorch 텐서를 NumPy 배열로 변환하여 OpenCV의 GaussianBlur를 직접 사용하는 함수.
        참고: 이 함수는 GPU->CPU 데이터 전송으로 인해 성능 저하가 발생할 수 있습니다.
        """

        # 1. 텐서 형태 확인 및 GPU 텐서를 CPU NumPy 배열로 변환
        if patch_tensor.dim() == 4 and patch_tensor.shape[0] == 1:
            patch_tensor = patch_tensor.squeeze(0) # [C, H, W]
        patch_tensor = self.unnorm(patch_tensor)

        # [C, H, W] -> [H, W, C] 형태로 변환하고 0-255 범위의 uint8 타입으로 변경
        img_np = patch_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 2. OpenCV를 사용하여 흑백 변환
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 3. OpenCV의 GaussianBlur 두 번 적용
        blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma1)
        blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma2)

        # 4. DoG 계산 (float으로 변환하여 오버플로우 방지)
        dog = blur1 - blur2
        
        # 5. NumPy를 사용하여 에너지 비율 계산
        e_hf = np.sum(dog**2)
        e_tot = np.sum(gray**2) + 1e-8
        hf_score = float(e_hf / e_tot)

        return hf_score

    def forward_feature(self, img, ref_dino, ref_clip, logit_size=None, ex_feats=None, last_feats=None, hf_score=None, num_sampled=None):
        h, w = img.shape[-2:]

        if ex_feats is None:
            ex_feats = self.ref_feature_dino(img)
        
        if type(img) == list:
            img = img[0]
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        elif last_feats is not None:
            image_features = self.net.encode_from_last_layer(
                last_feats, h, w, self.model_type, self.ignore_residual, output_cls_token=False, 
                ex_feats=ex_feats, ref_dino=ref_dino, ref_clip=ref_clip,
                hf_score=hf_score, num_sampled=num_sampled
            )
        else:
            image_features = self.net.encode_image(
                img, self.model_type, self.ignore_residual, output_cls_token=False, 
                ex_feats=ex_feats, ref_dino=ref_dino, ref_clip=ref_clip, 
                hf_score=hf_score, num_sampled=num_sampled
            ) 
        
        if self.output_cls_token:
            # image_cls_token, image_features = image_features
            # image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
            # cls_logits = image_cls_token @ self.query_features.T
            cls_logits = self.compute_cls_logits(img)

        # featup
        if self.feature_up:
            feature_h, feature_w = img[0].shape[-2] // self.dino_patch_size, img[0].shape[-1] // self.dino_patch_size
            image_h, image_w = img[0].shape[-2], img[0].shape[-1]
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_h, feature_w)
            with torch.cuda.amp.autocast():
                # --- Upsample with refinement ---
                if (feature_h, feature_w) == (14, 14):
                    image_features = self.upsampler.up2(image_features, img).half() # [1, 512, 28, 28]

                image_features = self.upsampler.up4(image_features, img).half() # [1, 512, 56, 56]]

                image_features = self.upsampler.up8(image_features, img).half() # [1, 512, 112, 112]

                image_features = self.upsampler.up16(image_features, img).half() # [1, 512, 224, 224]
        
                image_features = self.upsampler.fixup(image_features).half()
                # -------------------------------- 

            image_features = image_features.view(1, self.feat_dim, image_h * image_w).permute(0, 2, 1)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T

        if self.output_cls_token:
            logits = logits + cls_logits * self.cls_token_lambda

        if self.feature_up:
            h, w = img[0].shape[-2], img[0].shape[-1]
        else:
            h, w = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]

        # for proxy and not featup only
        # if self.vfm_model is not None:
        #     logits = logits.permute(0, 2, 1).reshape(-1, out_dim, I, J)
        # else:
        #     logits = logits.permute(0, 2, 1).reshape(-1, out_dim, h, w)
        # Original
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, h, w)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')

        return logits

    def ref_feature_dino(self, img, logit_size=None):
        clip_token_size = img.shape[-2] // self.net.visual.patch_size[0], img.shape[-1] // self.net.visual.patch_size[1]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)

        imgs_norm = imgs_norm.half()

        if self.vfm_model == 'sam':
            patch_size = self.vfm.image_encoder.patch_embed.proj.kernel_size
            imgs_norm = F.interpolate(imgs_norm, size=(1024, 1024), mode='bilinear', align_corners=False)
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            ex_feats = self.vfm.image_encoder(imgs_norm)
        elif self.vfm_model == 'dino':
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)

            # Forward pass in the model
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]

            nb_im = feat.shape[0]  # Batch size
            nb_tokens = feat.shape[1]  # Number of tokens
            nh = self.vfm.blocks[0].attn.num_heads  # Number of heads

            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]

            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size

            # ex_feats = q.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            # ex_feats = k.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            # ex_feats = v.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            ex_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
        elif self.vfm_model == 'dinov2':
            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-1] // patch_size[1]
            ex_feats = self.vfm.get_intermediate_layers(imgs_norm, reshape=True)[0]
        elif self.vfm_model == 'mae':
            patch_size = self.vfm.patch_embed.patch_size
            imgs_norm = F.interpolate(imgs_norm, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            image_feat = self.vfm.forward_features(imgs_norm)
            ex_feats = rearrange(image_feat, 'b (h w) c -> b c h w', h=I, w=J)
        else:
            I, J = clip_token_size
            ex_feats = None
        
        h_dino, w_dino = img.shape[-2] // self.dino_patch_size, img.shape[-1] // self.dino_patch_size
        ex_feats = F.interpolate(ex_feats, size=(h_dino, w_dino), mode='bilinear', align_corners=False)
        return ex_feats

    def _predict_feature_map(self, patch_img, x, crop_dino):
        """단일 패치에 대한 '피처맵' 단위의 분할 예측을 반환하는 헬퍼 함수"""

        image_features = self.net.encode_from_last_layer(x, self.model_type, self.ignore_residual, output_cls_token=False, ex_feats=crop_dino, ref_dino=None, ref_clip=None)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T # [1, 784, n_query]
        out_dim = logits.shape[-1]

        H_pad, W_pad = patch_img.shape[-2:]
        H_feat_pad, W_feat_pad = H_pad // self.dino_patch_size, W_pad // self.dino_patch_size

        seg_logits = logits.permute(0, 2, 1).reshape(-1, out_dim, H_feat_pad, W_feat_pad)
        seg_logits = seg_logits[0] * self.logit_scale
        seg_logits = seg_logits.softmax(0)  # n_queries * w * h
        num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
        if num_cls != num_queries:
            seg_logits = seg_logits.unsqueeze(0)
            cls_index = nn.functional.one_hot(self.query_idx)
            cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
            seg_logits = (seg_logits * cls_index).max(1)[0]
        seg_pred = seg_logits.argmax(0, keepdim=True)
        seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

        return seg_pred.squeeze(0) # [28, 28]

    # Best
    def forward_slide(self, img, img_metas, stride=112, crop_size=224, context_mode='global_only'):
        """
        [MODIFIED] Adds a `context_mode` option to control the context embeddings.
        Options for context_mode:
        - 'dino_proxy': No additonal embeddings.
        - 'global_only': Uses only global embeddings.
        - 'sampling_only': Uses only sampled embeddings.
        - 'sampling_and_global': Uses both sampled and global embeddings. (No gating)
        - 'gating': Single-scale gating for global embeddings.
        - 'multiscale_gating': Multi-scale gating for global embeddings.
        """
        printing = False

        if self.context_mode is not None:
            context_mode = self.context_mode

        use_global = 'global' in context_mode
        use_sampling = 'sampling' in context_mode
        use_gating = 'gating' in context_mode
        use_multiscale_gating = 'multiscale' in context_mode
        if 'gating' in context_mode:
            use_global = True
            use_sampling = True

        if printing:
            print("Printing the context mode:")
            print(f" - Using inter-region information: {use_sampling}")
            print(f" - Using global information: {use_global}")
            print(f" - Using gating for global information: {use_gating}")
            print(f" - Using multi-scale gating for global information: {use_multiscale_gating}")

        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)
        
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        device = img.device
        
        sample = img_metas[0]
        img_path = sample['img_path']
        basename = osp.basename(img_path)
        filename_without_ext = osp.splitext(basename)[0]
        embedding_path = osp.join(self.embedding_dir, f"{filename_without_ext}.pt")
        
        try:
            embeddings = torch.load(embedding_path, map_location=img.device)
            
            # 불러온 텐서에 배치 차원(unsqueeze) 추가
            loaded_global_dino_feats_flat = embeddings['global_dino_feats_flat'].to(dtype=torch.float32)
            loaded_global_clip_feats_flat = embeddings['global_clip_feats_flat'].to(dtype=torch.float32)
            loaded_all_robust_dino_feats  = embeddings['all_robust_dino_feats'].to(dtype=torch.float32)
            loaded_all_robust_clip_feats  = embeddings['all_robust_clip_feats'].to(dtype=torch.float32)
            
            # all_robust_patch_ids는 ID이므로 데이터 타입 변경 불필요
            loaded_all_robust_patch_ids   = embeddings['all_robust_patch_ids']
            
            # all_last_feats는 리스트이므로, 리스트 내 각 텐서의 데이터 타입을 변경
            loaded_all_last_feats = [t.to(device=img.device) for t in embeddings['all_last_feats']]

            loaded_inputs = embeddings['__debug_inputs'] # 저장된 inputs 텐서 로드
            print("Step B: Done.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}. "
                                    f"Please run extract_embeddings.py first.")

        # ========================================================================
        # Phase 0: Extract global features if needed
        # ========================================================================
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
                h_gl_tok, w_gl_tok = global_view_img.shape[-2] // self.patch_size[0], global_view_img.shape[-1] // self.patch_size[1]

                x_global = self.net.encode_before_last_layer(global_view_img)
                global_dino_feats = self.ref_feature_dino(global_view_img)
                global_clip_feats = self.net.encode_value_projection(x_global, h_gl_tok, w_gl_tok, target_size=(global_dino_feats.shape[-2:]))

                # dino_patch_size = self.dino_patch_size
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
                        pad = self.compute_padsize(H_orig, W_orig, self.patch_size[0])

                        if any(pad):
                            padded_img = F.pad(crop_img, pad, mode='constant', value=0)
                        else:
                            padded_img = crop_img

                        x = self.net.encode_before_last_layer(padded_img)
                        all_last_feats.append(x)

                        h_pad_tok, w_pad_tok = padded_img.shape[-2] // self.patch_size[0], padded_img.shape[-1] // self.patch_size[1]

                        dino_feat = self.ref_feature_dino(padded_img) 
                        target_h, target_w = dino_feat.shape[-2:]
                        clip_feat = self.net.encode_value_projection(x, h_pad_tok, w_pad_tok, target_size=(target_h, target_w))

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


        print("\nStep C: Comparing all tensors...")
        
        tensors_to_compare = {
            "global_dino_feats_flat": (global_dino_feats_flat, loaded_global_dino_feats_flat),
            "global_clip_feats_flat": (global_clip_feats_flat, loaded_global_clip_feats_flat),
            "all_robust_dino_feats": (all_robust_dino_feats, loaded_all_robust_dino_feats),
            "all_robust_clip_feats": (all_robust_clip_feats, loaded_all_robust_clip_feats),
            "all_robust_patch_ids": (all_robust_patch_ids, loaded_all_robust_patch_ids),
        }

        all_match = True
        for name, (realtime_t, loaded_t) in tensors_to_compare.items():
            print(f"--- Comparing: {name} ---")
            shape_match = realtime_t.shape == loaded_t.shape
            dtype_match = realtime_t.dtype == loaded_t.dtype
            # atol(absolute tolerance)은 float16/32의 미세한 오차를 감안하기 위해 필요합니다.
            value_match = torch.allclose(realtime_t.float(), loaded_t.float(), atol=1e-5)
            
            print(f"  Shape Match: {shape_match} ({realtime_t.shape} vs {loaded_t.shape})")
            print(f"  Dtype Match: {dtype_match} ({realtime_t.dtype} vs {loaded_t.dtype})")
            print(f"  Value Match: {value_match}")
            
            if not (shape_match and dtype_match and value_match):
                all_match = False

        # `all_last_feats` (리스트) 비교
        print("--- Comparing: all_last_feats (list) ---")
        if len(all_last_feats) != len(loaded_all_last_feats):
            print(f"  Length Mismatch: {len(all_last_feats)} vs {len(loaded_all_last_feats)}")
            all_match = False
        else:
            for i, (realtime_t, loaded_t) in enumerate(zip(all_last_feats, loaded_all_last_feats)):
                if not torch.allclose(realtime_t.float(), loaded_t.float(), atol=1e-5):
                    print(f"  Value Mismatch at index {i}")
                    all_match = False
                    break
            else:
                print("  Values Match!")

        print("\n========================================================")
        if all_match:
            print("✅ SUCCESS: All real-time and loaded tensors are identical!")
        else:
            print("❌ FAILURE: Discrepancy found between tensors!")
        print("========================================================\n")
        
        # --- [C] 두 `inputs` 텐서 직접 비교 ---
        print("\nStep C: Comparing preprocessed `inputs` tensors...")
        shape_match = img.shape == loaded_inputs.shape
        dtype_match = img.dtype == loaded_inputs.dtype
        value_match = torch.allclose(img, loaded_inputs, atol=1e-5)

        print(f"  Shape Match: {shape_match} ({img.shape} vs {loaded_inputs.shape})")
        print(f"  Dtype Match: {dtype_match} ({img.dtype} vs {loaded_inputs.dtype})")
        print(f"  Value Match: {value_match}")

        print("\n========================================================")
        if not value_match:
            print("🔥 ROOT CAUSE IDENTIFIED: Preprocessed image tensors are DIFFERENT.")
            print("   This confirms the issue is in the data loading/preprocessing pipeline.")
            diff = torch.abs(img.float() - loaded_inputs.float())
            print(f"   Max difference: {torch.max(diff)}")
            print(f"   Mean difference: {torch.mean(diff)}")
        else:
            print("✅ SUCCESS: Preprocessed inputs are IDENTICAL. The issue is NOT in the pipeline.")
        print("========================================================\n")
        # =================================================================
        #                    [DEBUGGING CODE END]
        # =================================================================



        # ========================================================================
        # Phase 2 & 3: Final prediction with selected context
        # ========================================================================
        if printing: print("Phase 2 & 3: Performing dynamic sampling and final prediction...")
        preds = img.new_zeros((batch_size, self.num_queries, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        patch_counter = 0

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # --- 현재 패치 crop 및 hf_score 계산 ---
                y1, x1 = h_idx * h_stride, w_idx * w_stride
                y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
                y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                if use_gating:
                    if use_multiscale_gating:
                        # hf_score, _ = self._calculate_hf_score_multiscale(crop_img)
                        hf_score, _ = self._calculate_hf_score_multiscale_torch(crop_img)
                    else:
                        # hf_score = self._calculate_hf_score(crop_img)
                        hf_score = self._calculate_hf_score_torch(crop_img)
                else:
                    hf_score = None
                
                sampled_dino, sampled_clip = None, None
                num_sampled = 0

                # --- Perform K-means sampling if needed ---
                if use_sampling and all_robust_dino_feats is not None:
                    external_mask = (all_robust_patch_ids != patch_counter)
                    external_dino_feats = all_robust_dino_feats[external_mask]
                    external_clip_feats = all_robust_clip_feats[external_mask]
                    num_external_robust = external_dino_feats.shape[0]

                    # internal_mask = (all_robust_patch_ids == patch_counter)
                    # internal_dino_feats = all_robust_dino_feats[internal_mask]
                    # #internal_dino_feats = internal_dino_feats.permute(1, 0).reshape(-1, 28, 28).unsqueeze(0) # <- 위험

                    if printing: 
                        print(f"  - Performing K-Means on {num_external_robust} feature embeddings.")

                    if num_external_robust > 0:
                        K = 25
                        if self.dino_patch_size == 16:
                            K = 10
                        K = min(K, num_external_robust)
                        kmeans = KMeans(n_clusters=K, init_method="kmeans++", mode="cosine", max_iter=25)
                        labels = kmeans.fit_predict(external_dino_feats)

                        M = 80
                        if self.dino_patch_size == 16:
                            M = 50
                        dino_samples, clip_samples = [], []
                        for cid in range(K):
                            member_indices = (labels == cid).nonzero(as_tuple=True)[0]
                            if len(member_indices) == 0: continue
                            
                            torch.manual_seed(42)
                            if len(member_indices) > M:
                                rand_indices = member_indices[torch.randperm(len(member_indices), device=device)[:M]]
                            else:
                                rand_indices = member_indices[torch.randint(0, len(member_indices), (M,), device=device)]
                            
                            dino_samples.append(external_dino_feats[rand_indices])
                            clip_samples.append(external_clip_feats[rand_indices])
                        
                    if dino_samples:
                        sampled_dino = torch.cat(dino_samples, dim=0)
                        sampled_clip = torch.cat(clip_samples, dim=0)
                        num_sampled = sampled_dino.shape[0]
                    
                # --- Construct final reference features based on context_mode ---
                dino_frags, clip_frags = [], []
                if use_sampling and sampled_dino is not None:
                    dino_frags.append(sampled_dino)
                    clip_frags.append(sampled_clip)
                if use_global and global_dino_feats_flat is not None:
                    dino_frags.append(global_dino_feats_flat)
                    clip_frags.append(global_clip_feats_flat)

                final_ref_dino = torch.cat(dino_frags, dim=0) if dino_frags else torch.empty(0, 768, device=device)
                final_ref_clip = torch.cat(clip_frags, dim=0) if clip_frags else torch.empty(0, 768, device=device)

                if printing: 
                    print(f"  - {final_ref_dino.shape[0]} feature embeddings in total.")

                ref_dino, ref_clip = None, None
                if final_ref_dino.shape[0] > 0:
                    ref_dino = final_ref_dino.t().unsqueeze(0).contiguous()
                    ref_clip = final_ref_clip.view(-1, self.num_head, self.head_dim).permute(1, 2, 0).contiguous()

                # --- Final Prediction ---
                y1, x1 = h_idx * h_stride, w_idx * w_stride
                y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
                y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                H_crop_orig, W_crop_orig = crop_img.shape[-2:]
                pad = self.compute_padsize(H_crop_orig, W_crop_orig, self.patch_size[0])
                if any(pad):
                    padded_crop_img = F.pad(crop_img, pad, mode='constant', value=0)
                else:
                    padded_crop_img = crop_img

                if not use_sampling:
                    crop_seg_logit = self.forward_feature(padded_crop_img, ref_dino, ref_clip, ex_feats=None, last_feats=None) #internal_dino_feats
                else:
                    crop_seg_logit = self.forward_feature(padded_crop_img, ref_dino, ref_clip, ex_feats=None, last_feats=all_last_feats[patch_counter], hf_score=hf_score, num_sampled=num_sampled) #internal_dino_feats

                # --- 예측 결과에서 패딩 제거 ---
                if any(pad):
                    l, _, t, _ = pad
                    # forward_feature의 출력은 패딩된 입력 크기(padded_crop_img)와 동일
                    # 따라서 원본 crop 크기(H_crop_orig, W_crop_orig)만큼 잘라냄
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H_crop_orig, l:l + W_crop_orig]

                preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1

                patch_counter += 1
        
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = F.interpolate(preds, size=img_size, mode='bilinear')

        return logits



    @torch.no_grad()
    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        inputs = inputs.half()
        # print(f"image size: {inputs.shape}")
        # print(data_samples[0].metainfo)
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)


    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
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

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices