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

@MODELS.register_module()
class ProxySegEarthSegmentation(BaseSegmentor):
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
        elif vfm_model == 'dinov2':
            # self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        elif vfm_model == 'dinov3':
            self.vfm = torch.hub.load('dinov3', 'dinov3_vitl16', source='local', weights='dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')
            # state_dict = torch.load('dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth', map_location='cpu')
            # # 만약 state_dict 안에 'model' key가 있다면 아래 라인의 주석을 해제하세요.
            # # state_dict = state_dict.get('model', state_dict)
            # self.vfm.load_state_dict(state_dict, strict=False) # strict=False는 일부 키가 달라도 유연하게 로드
            print("DINOv3 model loaded successfully.")
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

    @torch.no_grad()
    def build_cdam(self, attn_logits: torch.Tensor,
                scales      : list[float],
                T: float = .1,
                P: float = .1,
                device = 'cuda') -> torch.Tensor:
        """
        Args
        ----
        attn_logits : (C, H, W)   # patch×class logit(softmax 前)
        scales      : [0.25, 0.37, …]  # Multi-scale 비율
        Returns
        -------
        weight      : (1+N, 1+N)  # N = H*W, 0행/열은 [CLS]용 zero
        """
        C, H, W = attn_logits.shape
        N       = H * W
        device  = attn_logits.device
        dtype   = torch.float16                      # 메모리 절약

        softmax = torch.nn.Softmax(dim=0)
        S_full  = softmax(attn_logits / T)           # (C,H,W)

        # --------------- JS-divergence util (메모리 friendly) ---------------
        def js_div(mat: torch.Tensor) -> torch.Tensor:
            """
            mat : (C, N)  (float16)
            return : (N, N)  similarity (= 1-JS)
            """
            #   KL(p||M)+KL(q||M) 를 vectorized 로 계산
            p  = mat.unsqueeze(2)                    # (C,N,1)
            q  = mat.unsqueeze(1)                    # (C,1,N)
            M  = 0.5 * (p + q)

            js = 0.5 * (
                (p * (p.add(1e-8).log() - M.add(1e-8).log())).sum(0) +
                (q * (q.add(1e-8).log() - M.add(1e-8).log())).sum(0)
            )                                        # (N,N)

            sim = 1. - js                            # (0~1) similarity
            sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
            sim = torch.softmax(sim / P, dim=1)      # row-softmax
            return sim.to(dtype)

        # (1) 원본 해상도
        S0_flat  = S_full.reshape(C, N).to(dtype)
        sims     = [js_div(S0_flat)]

        # (2) Multi-scale (down→prob→up 후 JS계산)
        for r in scales:
            h2, w2 = max(1, int(H*r)), max(1, int(W*r))
            # ① logits ↓
            log_r = F.interpolate(attn_logits.unsqueeze(0), size=(h2, w2),
                                mode='bilinear', align_corners=False)[0]
            # ② softmax → prob ↓
            S_r  = softmax(log_r / T)                # (C,h2,w2)
            # ③ 다시 원본 크기로 ↑ (C,H,W)  --> patch index 정합성 확보
            S_r  = F.interpolate(S_r.unsqueeze(0), size=(H, W),
                                mode='bilinear', align_corners=False)[0]
            sims.append(js_div(S_r.reshape(C, N)))

        # (3) 평균 결합
        A = torch.stack(sims, 0).mean(0)             # (N,N)

        # (4) weight 매트릭스 (CLS 행/열=0)
        weight = torch.zeros((N+1, N+1), device=device, dtype=dtype)
        weight[1:, 1:] = A
        return weight

    def forward_feature(self, img, logit_size=None):
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
        elif self.vfm_model == 'dinov3':
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
        
        # # --- DINO attention ---
        # beta = 1.2
        # gamma = 3.0
        # temperature = 0.3
        # lam = 1.0

        # ex_feats_2 = F.interpolate(ex_feats, size=(2*ex_feats.shape[-2], 2*ex_feats.shape[-1]),
        #                             mode='bilinear', align_corners=False)
        # # ex_feats_3 = F.interpolate(ex_feats, size=(4*ex_feats.shape[-2], 4*ex_feats.shape[-1]),
        # #                             mode='bilinear', align_corners=False)
        # # ex_feats_4 = F.interpolate(ex_feats, size=(8*ex_feats.shape[-2], 8*ex_feats.shape[-1]),
        # #                             mode='bilinear', align_corners=False)

        # def get_attn(feats):
        #     q_k = F.normalize(feats.flatten(2, 3), dim=1) # [1, 768, 784]
        #     similarity = torch.einsum("b c m, b c n -> b m n", q_k, q_k) # [1, 784, 784]
            
        #     similarity = (similarity - torch.mean(similarity) * beta) * gamma
        #     similarity[similarity < 0.0] = float('-inf')

        #     attn_weights = F.softmax(similarity/temperature, dim=-1) # [1, 784, 784]
        #     return attn_weights

        # # attn_weights = get_attn(ex_feats)
        # attn_weights_2 = get_attn(ex_feats_2)
        # # attn_weights_3 = get_attn(ex_feats_3)
        # # attn_weights_4 = get_attn(ex_feats_4)

        # # --- Dino attention ---

        if type(img) == list:
            img = img[0]
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        else:
            image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, output_cls_token=False, ex_feats=ex_feats)#self.output_cls_token)
            


        # # ---------------CDAM------------------
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # logits = image_features @ self.query_features.T

        # w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        # out_dim = logits.shape[-1]

        # # (shape 변환) → (C,H,W)  / 패치 해상도 계산
        # logits_map = logits.squeeze(0).permute(1,0).view(-1, w, h)  # (C,H,W)

        # # ────────────── (B) CDAM attention 계산 ──────────────
        # scales = [0.25,0.37,0.5,0.63,0.75,0.87]
        # weight = self.build_cdam(logits_map, scales, T=0.1, P=0.1, device=logits_map.device)

        # # ────────────── (C) 마지막 Transformer layer에 삽입 ──────────────
        # for blk in self.net.visual.transformer.resblocks[:-1]:
        #     x = blk(x)
        # for blk in self.net.visual.transformer.resblocks[-1:]:     # 1개 block
        #     x = x + self.net.visual.custom_dual_attn(
        #                 blk.attn, blk.ln_1(x), weight=weight, csa=True)
        #     x = x + blk.mlp(blk.ln_2(x))

        # # (D) CDAM 이후 feature → class logits 재계산
        # x = x.permute(1,0,2)                                   # (NLD)→(N,L,D)
        # image_features = self.net.visual.ln_post(x) @ self.net.visual.proj
        # image_features = image_features[:,1:]                  # [CLS] 제외
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # logits = image_features @ self.query_features.T

        # logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        # # ---------------CDAM------------------



        
        if self.output_cls_token:
            # image_cls_token, image_features = image_features
            # image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
            # cls_logits = image_cls_token @ self.query_features.T
            cls_logits = self.compute_cls_logits(img)

        # featup
        if self.feature_up:
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            # image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, 28, 28)
            with torch.cuda.amp.autocast():
                # image_features = F.interpolate(image_features, size=(56, 56),
                #                       mode='bilinear', align_corners=False)
                #image_features = self.upsampler(image_features, img).half()
                # --- Upsample with refinement ---
                # image_features = self.upsampler.up2(image_features, img).half() # [1, 512, 28, 28]
                # # refine 1
                # image_features = image_features.view(1, self.feat_dim, -1)
                # refine1 = torch.einsum("b c n, b q n -> b c q", image_features, attn_weights) # [1, 512, 784]
                # image_features = (refine1 + lam * image_features).view(1, self.feat_dim, 2*feature_w, 2*feature_h)

                image_features = self.upsampler.up4(image_features, img).half() # [1, 512, 56, 56]
                # # refine 2
                # image_features = image_features.view(1, self.feat_dim, -1)
                # refine2 = torch.einsum("b c n, b q n -> b c q", image_features, attn_weights_2)
                # image_features = (refine2 + lam * image_features).view(1, self.feat_dim, 4*feature_w, 4*feature_h)

                image_features = self.upsampler.up8(image_features, img).half() # [1, 512, 112, 112]
                # # refine 3
                # image_features = image_features.view(1, self.feat_dim, -1)
                # refine3 = torch.einsum("b c n, b q n -> b c q", image_features, attn_weights_3)
                # image_features = (refine3 + lam * image_features).view(1, self.feat_dim, 8*feature_w, 8*feature_h)

                image_features = self.upsampler.up16(image_features, img).half() # [1, 512, 224, 224]
                # refine 4
                # image_features = image_features.view(1, self.feat_dim, -1)
                # refine4 = torch.einsum("b c n, b q n -> b c q", image_features, attn_weights_4)
                # image_features = (refine4 + lam * image_features).view(1, self.feat_dim, 16*feature_w, 16*feature_h)

                image_features = self.upsampler.fixup(image_features).half()
                # -------------------------------- 
            image_features = image_features.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T

        if self.output_cls_token:
            logits = logits + cls_logits * self.cls_token_lambda

            # # CLIP Surgery
            # # weights to restrain influence of obvious classes on others
            # prob = (cls_logits * 2).softmax(-1)
            # w = prob / prob.mean(-1, keepdim=True)
            # # element-wise multiplied features
            # b, n_t, n_i, c = image_features.shape[0], self.query_features.shape[0], image_features.shape[1], image_features.shape[2]
            # feats = image_features.reshape(b, n_i, 1, c) * self.query_features.reshape(1, 1, n_t, c)
            # feats *= w.reshape(1, 1, n_t, 1)
            # redundant_feats = feats.mean(2, keepdim=True) # along cls dim
            # feats = feats - redundant_feats
            # # sum the element-wise multiplied features as cosine similarity
            # logits = feats.sum(-1)

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

        return logits

    def forward_feature_class(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        else:
            image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, self.output_cls_token)
            
        image_cls_token, image_features = image_features
        image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
        cls_logits = image_cls_token @ self.query_features.T

        # featup
        if self.feature_up:
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            with torch.cuda.amp.autocast():
                image_features = self.upsampler(image_features, img).half()
            image_features = image_features.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T

        if self.output_cls_token:
            logits = logits + cls_logits * self.cls_token_lambda

            # # CLIP Surgery
            # # weights to restrain influence of obvious classes on others
            # prob = (cls_logits * 2).softmax(-1)
            # w = prob / prob.mean(-1, keepdim=True)
            # # element-wise multiplied features
            # b, n_t, n_i, c = image_features.shape[0], self.query_features.shape[0], image_features.shape[1], image_features.shape[2]
            # feats = image_features.reshape(b, n_i, 1, c) * self.query_features.reshape(1, 1, n_t, c)
            # feats *= w.reshape(1, 1, n_t, 1)
            # redundant_feats = feats.mean(2, keepdim=True) # along cls dim
            # feats = feats - redundant_feats
            # # sum the element-wise multiplied features as cosine similarity
            # logits = feats.sum(-1)

        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')

        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        # def build_patch_weight(device, base_size=224, grid=4):
        #     """
        #     Build a 1×1×224×224 weight map:
        #         centre 4 grids   → weight 4
        #         edge   8 grids   → weight 2
        #         corner 4 grids   → weight 1
        #     """
        #     cell = base_size // grid          # 56 for 224 / 4
        #     w = torch.ones(1, 1, base_size, base_size, device=device)   # default 1 (corners)

        #     for i in range(grid):
        #         for j in range(grid):
        #             h0, h1 = i*cell, (i+1)*cell
        #             w0, w1 = j*cell, (j+1)*cell
        #             if 1 <= i <= 2 and 1 <= j <= 2:               # centre 2×2 cells (4 total)
        #                 w[..., h0:h1, w0:w1] = 9
        #             elif (1 <= i <= 2) ^ (1 <= j <= 2):            # edges (but not centre, not corner)
        #                 w[..., h0:h1, w0:w1] = 3
        #             # else keep 1 (corners)
        #     return w                                             # shape (1,1,224,224)
        
        # weight_template = build_patch_weight(img.device)        # (1,1,224,224)        

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                # # ------------------------------------------
                # patch_weight = weight_template[..., :crop_seg_logit.size(2), :crop_seg_logit.size(3)]
                # patch_weight = patch_weight.expand(-1, crop_seg_logit.size(1), -1, -1)
                # # apply weights
                # weighted_logit  = crop_seg_logit * patch_weight          # [B,Q,H,W]

                # # 3. paste the weighted logits into the full-image canvas
                # preds += nn.functional.pad(weighted_logit,
                #                         (int(x1), int(preds.shape[3] - x2),
                #                             int(y1), int(preds.shape[2] - y2)))

                # # 4. accumulate the weights (not just “+ 1”) for later normalisation
                # count_mat += nn.functional.pad(patch_weight[:, :1, ...],      # keep 1 channel
                #                             (int(x1), int(count_mat.shape[3] - x2),
                #                                 int(y1), int(count_mat.shape[2] - y2)))
                # # ------------------------------------------

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

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

        # def rotate_img_metas(batch_img_metas, swap_hw=False):
        #     """
        #     batch_img_metas: list of dicts (각 이미지의 meta 정보)
        #     swap_hw: True이면 height/width를 swap
        #     """
        #     rotated_metas = []
        #     for meta in batch_img_metas:
        #         meta_copy = meta.copy()
        #         for key in ['ori_shape', 'img_shape', 'pad_shape']:
        #             if key in meta_copy and isinstance(meta_copy[key], tuple) and len(meta_copy[key]) >= 2:
        #                 if swap_hw:
        #                     meta_copy[key] = (meta_copy[key][1], meta_copy[key][0], *meta_copy[key][2:])
        #         rotated_metas.append(meta_copy)
        #     return rotated_metas

        # seg_logits_list = []
        # # 회전 각도별 처리 (0, 90, 180, 270)
        # for k in range(4):
        #     rotated_inputs = torch.rot90(inputs, k=k, dims=(2, 3))

        #     # 90°, 270° 회전 시 swap_hw=True
        #     swap_hw = (k % 2 == 1)
        #     rot_img_metas = rotate_img_metas(batch_img_metas, swap_hw=swap_hw)

        #     if self.slide_crop > 0:
        #         seg_logits_rot = self.forward_slide(rotated_inputs, rot_img_metas,
        #                                             self.slide_stride, self.slide_crop)
        #     else:
        #         seg_logits_rot = self.forward_feature(rotated_inputs, rot_img_metas[0]['ori_shape'])

        #     # 원래 방향으로 되돌림 (inverse rotation)
        #     if k > 0:
        #         seg_logits_rot = torch.rot90(seg_logits_rot, k=4-k, dims=(2, 3))

        #     seg_logits_list.append(seg_logits_rot)
        # # 평균

        # resized_inputs = F.interpolate(inputs, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
        # seg_logits_global = self.forward_feature(resized_inputs, logit_size=batch_img_metas[0]['ori_shape'])
        # seg_logits_list.append(seg_logits_global)
        # # seg_logits_list.append(seg_logits_global)

        # seg_logits = torch.stack(seg_logits_list, dim=0).mean(dim=0)





        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)

            resized_inputs = F.interpolate(inputs, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            seg_logits_global = self.forward_feature(resized_inputs, logit_size=batch_img_metas[0]['ori_shape'])
            seg_logits = (seg_logits + seg_logits_global) / 2.0
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)

    @torch.no_grad()
    def predict_prob(self, inputs, data_samples):
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
        # seg_logits: [B, Q, H, W]   (Q = n_queries)

        probs_batch = []               # 최종 [B, C, H, W] 리스트
        num_cls = max(self.query_idx) + 1
        num_queries = len(self.query_idx)

        for b in range(seg_logits.shape[0]):
            # (a) logit scaling + softmax over query dim
            q_prob = (seg_logits[b] * self.logit_scale).softmax(dim=0)  # [Q,H,W]

            # (b) query → class 집계
            if num_cls != num_queries:
                # one-hot mask: [C,Q,1,1]
                cls_mask = F.one_hot(self.query_idx,
                                    num_classes=num_cls).T[..., None, None]
                # [C,Q,H,W] → max over query dim
                c_prob = (q_prob.unsqueeze(0) * cls_mask).max(dim=1)[0] # [C,H,W]
            else:
                c_prob = q_prob                                        # [C,H,W]

            # (c) 클래스별 확률 정규화  (합 = 1)
            c_prob = c_prob / c_prob.sum(dim=0, keepdim=True).clamp(min=1e-8)

            # (d) 선택 옵션: 배경 임계값 처리
            if hasattr(self, 'prob_thd'):
                low_conf = c_prob.max(dim=0, keepdim=True)[0] < self.prob_thd
                c_prob[:, low_conf.squeeze(0)] = 0.0
                c_prob[self.bg_idx, low_conf.squeeze(0)] = 1.0

            probs_batch.append(c_prob)   # [C,H,W]

            # (e) MMSeg data_sample 저장 (선택)
            if data_samples is not None:
                seg_pred = c_prob.argmax(dim=0, keepdim=True)  # [1,H,W]
                data_samples[b].set_data({
                    'pred_prob':   PixelData(**{'data': c_prob}),
                    'pred_sem_seg': PixelData(**{'data': seg_pred})
                })

        # ---- 4. 반환 -----------------------------------------------------------
        if data_samples is None:
            return torch.stack(probs_batch, dim=0)     # [B,C,H,W]
        return data_samples

    @torch.no_grad()
    def predict_logit(self, inputs: torch.Tensor, data_samples=None):
        """Return per-class aggregated logits (no softmax) for each pixel.

        Args:
            inputs (Tensor): BCHW image tensor.
            data_samples (list[SegDataSample] | None): optional MMSeg samples.

        Returns:
            Tensor | list[SegDataSample]:
                * if ``data_samples is None`` → Tensor [B, C, H, W] of logits
                * else → ``data_samples`` with ``pred_logit`` (PixelData)
        """
        # ------------------------------------------------------------------ #
        # 0) meta 정보 확보 (predict / predict_prob 와 동일)
        # ------------------------------------------------------------------ #
        if data_samples is not None:
            batch_img_metas = [ds.metainfo for ds in data_samples]
        else:
            batch_img_metas = [dict(
                ori_shape=inputs.shape[2:], img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:], padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        inputs = inputs.half()

        # ------------------------------------------------------------------ #
        # 1) forward (slide or whole) → seg_logits : [B, Q, H, W]
        # ------------------------------------------------------------------ #
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas,
                                            self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs,
                                            batch_img_metas[0]['ori_shape'])

        # ------------------------------------------------------------------ #
        # 2) query → class 집계 (logit 수준)                                  #
        # ------------------------------------------------------------------ #
        logit_batch = []                                  # [B, C, H, W]
        num_cls     = max(self.query_idx) + 1
        num_queries = len(self.query_idx)

        for b in range(seg_logits.shape[0]):
            q_logit = seg_logits[b]                       # [Q,H,W]

            if num_cls != num_queries:
                # one-hot mask : [C,Q,1,1]
                cls_mask = F.one_hot(
                    self.query_idx, num_classes=num_cls
                ).T.float()[..., None, None]              # [C,Q,1,1]

                # [C,Q,H,W] → max over Q : [C,H,W]
                c_logit = (q_logit.unsqueeze(0) * cls_mask).max(dim=1)[0]
            else:
                # 1:1 매핑이면 그대로
                c_logit = q_logit                         # [C,H,W]

            logit_batch.append(c_logit)

            # ------------------------------------------------------------------ #
            # 3) MMSeg data_sample 에 저장 (선택)
            # ------------------------------------------------------------------ #
            if data_samples is not None:
                # 필요하다면 여기서 softmax/argmax 도 추가로 계산 가능
                data_samples[b].set_data({
                    'pred_logit': PixelData(**{'data': c_logit})
                })

        # ------------------------------------------------------------------ #
        # 4) 반환
        # ------------------------------------------------------------------ #
        if data_samples is None:
            return torch.stack(logit_batch, dim=0)        # [B,C,H,W]
        return data_samples

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