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
import numpy as np

from myutils import UnNormalize
from segment_anything import sam_model_registry
# from sklearn.cluster import KMeans
from fast_pytorch_kmeans import KMeans
import torchvision.transforms.functional as TF

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

    def forward_feature(self, img, ref_dino, ref_clip, logit_size=None, ex_feats=None, last_feats=None):
        
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
            image_features = self.net.encode_from_last_layer(last_feats, self.model_type, self.ignore_residual, output_cls_token=False, ex_feats=ex_feats, ref_dino=ref_dino, ref_clip=ref_clip)
        else:
            image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, output_cls_token=False, ex_feats=ex_feats, ref_dino=ref_dino, ref_clip=ref_clip)#self.output_cls_token)  
        
        if self.output_cls_token:
            # image_cls_token, image_features = image_features
            # image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
            # cls_logits = image_cls_token @ self.query_features.T
            cls_logits = self.compute_cls_logits(img)

        # featup
        if self.feature_up:
            #feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            feature_w, feature_h = img[0].shape[-2] // self.dino_patch_size, img[0].shape[-1] // self.dino_patch_size
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            # image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            with torch.cuda.amp.autocast():
                # --- Upsample with refinement ---
                # image_features = self.upsampler.up2(image_features, img).half() # [1, 512, 28, 28]

                image_features = self.upsampler.up4(image_features, img).half() # [1, 512, 56, 56]]

                image_features = self.upsampler.up8(image_features, img).half() # [1, 512, 112, 112]

                image_features = self.upsampler.up16(image_features, img).half() # [1, 512, 224, 224]
        
                image_features = self.upsampler.fixup(image_features).half()
                # -------------------------------- 

                smoothing = True
                if smoothing:
                    k = 3
                    window_size = 2 * k + 1
                    temperature = 0.01
                    B, C, H, W = image_features.shape

                    # 1. unfold -> window
                    original_patches = F.unfold(image_features, kernel_size=window_size, padding=k) # [B, C * window_size^2, H*W]
                    original_patches = original_patches.view(B, C, window_size**2, H * W)

                    # 2. Normalized features for softmax
                    normalized_patches = F.normalize(original_patches, dim=1)

                    # 2. Cosine similarity
                    center_idx = window_size**2 // 2
                    center_features_norm = normalized_patches[:, :, center_idx:center_idx+1, :].expand_as(normalized_patches)
                    similarity = torch.sum(center_features_norm * normalized_patches, dim=1)

                    # 3. Softmax for weight
                    # [B, H*W, window_size^2] 
                    weights = F.softmax(similarity.permute(0, 2, 1) / temperature, dim=-1)

                    # 4. Weighted Average
                    # weights: [B, H*W, window_size^2] -> [B, 1, H*W, window_size^2]
                    # original_patches: [B, C, window_size^2, H*W] -> [B, C, H*W, window_size^2]
                    weighted_avg = torch.sum(
                        original_patches.permute(0, 1, 3, 2) * weights.unsqueeze(1),
                        dim=-1
                    ) # shape: [B, C, H*W]

                    # 5. Final process
                    image_features = weighted_avg.view(B, C, H, W).half()

            image_features = image_features.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T

        if self.output_cls_token:
            logits = logits + cls_logits * self.cls_token_lambda

        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]

        # for proxy and not featup only
        # if self.vfm_model is not None:
        #     logits = logits.permute(0, 2, 1).reshape(-1, out_dim, I, J)
        # else:
        #     logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        # Original
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')

        smoothing = False
        if smoothing:
            seg_logits_item = logits[0]
            device = seg_logits_item.device

            k = 4
            h = k**2 + 2 * k  
            window_size = 2 * k + 1  
            C, H, W = seg_logits_item.shape

            # 1. logit -> probs for JSD
            probs = (seg_logits_item * self.logit_scale).softmax(dim=0)

            # 2. unfold 함수로 슬라이딩 윈도우 생성 (확률 및 로짓 모두)
            padding = k  # 이미지 경계 처리를 위한 패딩
            prob_patches = F.unfold(
                probs.unsqueeze(0),
                kernel_size=window_size,
                padding=padding
            ).squeeze(0)
            # [C, 윈도우크기*윈도우크기, H*W] 형태로 변경
            prob_patches = prob_patches.view(C, window_size**2, H * W)

            logit_patches = F.unfold(
                seg_logits_item.unsqueeze(0),
                kernel_size=window_size,
                padding=padding
            ).squeeze(0)
            logit_patches = logit_patches.view(C, window_size**2, H * W)

            # 3. 중심 픽셀과 이웃 픽셀 간의 JSD 계산
            center_probs = probs.view(C, H * W).unsqueeze(1).expand(-1, window_size**2, -1)
            
            m = 0.5 * (center_probs + prob_patches)
            eps = 1e-10  # log(0) 방지를 위한 작은 값

            # F.kl_div(input, target)은 D_KL(target || input)을 계산하며, input으로 log-prob를 기대함
            log_m = torch.log(m + eps)
            jsd_scores = 0.5 * (
                F.kl_div(log_m, center_probs, reduction='none', log_target=False).sum(dim=0) +
                F.kl_div(log_m, prob_patches, reduction='none', log_target=False).sum(dim=0)
            )
            # jsd_scores의 최종 shape: [윈도우크기*윈도우크기, H*W]

            # 4. JSD가 가장 낮은 (가장 유사한) 상위 h개 이웃 선택
            top_h_indices = torch.topk(jsd_scores, h, dim=0, largest=False).indices
            # top_h_indices의 shape: [h, H*W]

            # 5. 선택된 이웃들의 원본 로짓을 가져오기
            expanded_indices = top_h_indices.unsqueeze(0).expand(C, -1, -1)
            selected_logits = torch.gather(logit_patches, 1, expanded_indices)
            # selected_logits의 shape: [C, h, H*W]

            # 6. 선택된 로짓들의 평균을 계산하여 새로운 로짓으로 사용
            averaged_logits = selected_logits.mean(dim=1)
            smoothed_logits = averaged_logits.view(C, H, W)
            # --- JSD 스무딩 로직 종료 ---

            logits = smoothed_logits.unsqueeze(0)

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
            I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-2] // patch_size

            # ex_feats = q.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            # ex_feats = k.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            # ex_feats = v.reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
            ex_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)
        elif self.vfm_model == 'dinov2':
            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
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
    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """
        [수정됨] 각 슬라이딩 패치에 대해 '자신을 제외한' 외부 정보로 클러스터링 및 샘플링 수행
        """
        printing = False

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
        
        # ========================================================================
        # Phase 1: 강건한(Robust) 개별 피처 임베딩 및 '패치 ID' 수집
        # ========================================================================
        if printing: print("Phase 1: Collecting robust feature embeddings and their patch IDs...") 
        
        all_robust_dino = []
        all_robust_clip = []
        all_robust_patch_ids = [] # [추가] 각 임베딩의 소속 패치 ID를 저장할 리스트
        total_feature_patches = 0
        patch_counter = 0

        all_last_feats = []

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

                dino_feat = self.ref_feature_dino(padded_img) # 원래는 여기 pad 안한 걸 썼음 (전달용)
                clip_feat = self.net.encode_value_projection(x)

                # seg_pred = self._predict_feature_map(padded_img, x, dino_feat)

                # if any(pad):
                #     H_feat_orig = H_orig // self.dino_patch_size
                #     W_feat_orig = W_orig // self.dino_patch_size
                #     l_feat, _, t_feat, _ = [p // self.dino_patch_size for p in pad]
            
                #     seg_pred = seg_pred[t_feat:t_feat + H_feat_orig, l_feat:l_feat + W_feat_orig]

                # H_feat, W_feat = seg_pred.shape
                # total_feature_patches += H_feat * W_feat

                dino_feat_flat = dino_feat.flatten(2, 3).permute(0, 2, 1).reshape(-1, dino_feat.shape[1])
                clip_feat_flat = clip_feat.flatten(2, 3).permute(2, 0, 1).reshape(-1, clip_feat.shape[0] * clip_feat.shape[1])

                all_robust_dino.append(dino_feat_flat)
                all_robust_clip.append(clip_feat_flat)

                # [추가] 이 강건한 피처들이 현재 패치(patch_counter) 소속임을 기록
                num_robust_in_patch = dino_feat_flat.shape[0]
                all_robust_patch_ids.append(torch.full((num_robust_in_patch,), patch_counter, device=device))

                patch_counter += 1
        
        # 수집된 모든 강건한 임베딩과 ID를 하나의 텐서로 통합
        if not all_robust_dino:
             all_robust_dino_feats = torch.empty(0, 768, device=device)
             all_robust_clip_feats = torch.empty(0, 768, device=device)
             all_robust_patch_ids = torch.empty(0, dtype=torch.long, device=device)
        else:
            all_robust_dino_feats = torch.cat(all_robust_dino, dim=0).to(dtype=torch.float32)
            all_robust_clip_feats = torch.cat(all_robust_clip, dim=0).to(dtype=torch.float32)
            all_robust_patch_ids = torch.cat(all_robust_patch_ids, dim=0)

        num_total_robust = all_robust_dino_feats.shape[0]
        if printing: print(f"  - Collected {num_total_robust} robust feature embeddings in total.")

        # ========================================================================
        # Phase 2 & 3: 동적 외부 정보 샘플링 및 최종 예측
        # ========================================================================
        if printing: print("Phase 2 & 3: Performing dynamic sampling and final prediction...")
        preds = img.new_zeros((batch_size, self.num_queries, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        patch_counter = 0

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # --- 1. 현재 패치에 대한 '외부 정보 풀' 구성 ---
                ref_dino, ref_clip = None, None
                
                # 현재 패치 ID를 제외한 외부 임베딩만 선택
                external_mask = (all_robust_patch_ids != patch_counter)
                external_dino_feats = all_robust_dino_feats[external_mask]
                external_clip_feats = all_robust_clip_feats[external_mask]

                internal_mask = (all_robust_patch_ids == patch_counter)
                internal_dino_feats = all_robust_dino_feats[internal_mask]
                #internal_dino_feats = internal_dino_feats.permute(1, 0).reshape(-1, 28, 28).unsqueeze(0) # <- 위험

                num_external_robust = external_dino_feats.shape[0]
                
                if num_external_robust > 0:
                    # --- 2. 외부 정보 풀에 대해 K-means 및 샘플링 수행 ---
                    K = 25
                    K = min(K, num_external_robust)
                    # kmeans = KMeans(n_clusters=K, init="k-means++", n_init=1, max_iter=25, random_state=42)
                    # labels = torch.tensor(kmeans.fit_predict(external_dino_feats.cpu().numpy()), device=device, dtype=torch.long)
                    kmeans = KMeans(n_clusters=K, init_method="kmeans++", mode="cosine", max_iter=25)
                    labels = kmeans.fit_predict(external_dino_feats)

                    M = 80
                    dino_samples, clip_samples = [], []
                    for cid in range(K):
                        member_indices = (labels == cid).nonzero(as_tuple=True)[0]
                        if len(member_indices) == 0: continue
                        
                        torch.manual_seed = 42
                        if len(member_indices) > M:
                            rand_indices = member_indices[torch.randperm(len(member_indices), device=device)[:M]]
                        else:
                            rand_indices = member_indices[torch.randint(0, len(member_indices), (M,), device=device)]
                        
                        dino_samples.append(external_dino_feats[rand_indices])
                        clip_samples.append(external_clip_feats[rand_indices])
                    
                    if dino_samples:
                        ref_dino = torch.cat(dino_samples, dim=0).t().unsqueeze(0).contiguous()
                        ref_clip = torch.cat(clip_samples, dim=0).view(-1, 12, 64).permute(1, 2, 0).contiguous()

                # --- 3. 최종 예측 수행 ---
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

                crop_seg_logit = self.forward_feature(padded_crop_img, ref_dino, ref_clip, ex_feats=None, last_feats=all_last_feats[patch_counter]) #internal_dino_feats

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
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)


    def postprocess_result(self, seg_logits, data_samples):
        smoothing = False

        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            if smoothing:
                seg_logits_item = seg_logits[i]
                device = seg_logits_item.device

                k = 4
                h = k**2 + 2 * k  
                window_size = 2 * k + 1  
                C, H, W = seg_logits_item.shape

                # 1. logit -> probs for JSD
                probs = (seg_logits_item * self.logit_scale).softmax(dim=0)

                # 2. unfold 함수로 슬라이딩 윈도우 생성 (확률 및 로짓 모두)
                padding = k  # 이미지 경계 처리를 위한 패딩
                prob_patches = F.unfold(
                    probs.unsqueeze(0),
                    kernel_size=window_size,
                    padding=padding
                ).squeeze(0)
                # [C, 윈도우크기*윈도우크기, H*W] 형태로 변경
                prob_patches = prob_patches.view(C, window_size**2, H * W)

                logit_patches = F.unfold(
                    seg_logits_item.unsqueeze(0),
                    kernel_size=window_size,
                    padding=padding
                ).squeeze(0)
                logit_patches = logit_patches.view(C, window_size**2, H * W)

                # 3. 중심 픽셀과 이웃 픽셀 간의 JSD 계산
                center_probs = probs.view(C, H * W).unsqueeze(1).expand(-1, window_size**2, -1)
                
                m = 0.5 * (center_probs + prob_patches)
                eps = 1e-10  # log(0) 방지를 위한 작은 값

                # F.kl_div(input, target)은 D_KL(target || input)을 계산하며, input으로 log-prob를 기대함
                log_m = torch.log(m + eps)
                jsd_scores = 0.5 * (
                    F.kl_div(log_m, center_probs, reduction='none', log_target=False).sum(dim=0) +
                    F.kl_div(log_m, prob_patches, reduction='none', log_target=False).sum(dim=0)
                )
                # jsd_scores의 최종 shape: [윈도우크기*윈도우크기, H*W]

                # 4. JSD가 가장 낮은 (가장 유사한) 상위 h개 이웃 선택
                top_h_indices = torch.topk(jsd_scores, h, dim=0, largest=False).indices
                # top_h_indices의 shape: [h, H*W]

                # 5. 선택된 이웃들의 원본 로짓을 가져오기
                expanded_indices = top_h_indices.unsqueeze(0).expand(C, -1, -1)
                selected_logits = torch.gather(logit_patches, 1, expanded_indices)
                # selected_logits의 shape: [C, h, H*W]

                # 6. 선택된 로짓들의 평균을 계산하여 새로운 로짓으로 사용
                averaged_logits = selected_logits.mean(dim=1)
                smoothed_logits = averaged_logits.view(C, H, W)
                # --- JSD 스무딩 로직 종료 ---

                # 7. 스무딩된 로짓을 사용하여 이후 후처리 과정 진행
                final_probs = (smoothed_logits * self.logit_scale).softmax(0)

                num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
                if num_cls != num_queries:
                    final_probs = final_probs.unsqueeze(0)
                    # num_classes를 명시하여 one_hot 인덱싱 오류 방지
                    cls_index = F.one_hot(self.query_idx, num_classes=num_cls).to(device)
                    cls_index = cls_index.T.view(num_cls, num_queries, 1, 1).type_as(final_probs)
                    # 각 클래스에 해당하는 쿼리 확률 중 가장 큰 값을 사용
                    final_probs = (final_probs * cls_index).max(1)[0]

                seg_pred = final_probs.argmax(0, keepdim=True)
                seg_pred[final_probs.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx
                
                if data_samples is None:
                    return seg_pred
                else:
                    data_samples[i].set_data({
                        'seg_logits':
                            PixelData(**{'data': seg_logits_item}),
                        'pred_sem_seg':
                            PixelData(**{'data': seg_pred})
                    })
            else:
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