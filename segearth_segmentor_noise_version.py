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

@MODELS.register_module()
class SegEarthSegmentation(BaseSegmentor):
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
        if vfm_model is None:
            self.vfm = None
        elif vfm_model.lower() == 'dino':
            # ViT‑B/8 gives the best spatial resolution while staying light‑weight
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').requires_grad_(False).eval().to(device)
        elif vfm_model.lower() == 'dinov2':
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').requires_grad_(False).eval().to(device)
        elif vfm_model.lower() == 'sam':
            from segment_anything import sam_model_registry            # type: ignore
            self.vfm = sam_model_registry['vit_b']().requires_grad_(False).eval().to(device)
        elif vfm_model.lower() == 'mae':
            from mae import models_vit                                            # type: ignore
            self.vfm = models_vit.__dict__['vit_base_patch16'](img_size=self.slide_crop, num_classes=0, global_pool=False)
            self.vfm = self.vfm.requires_grad_(False).eval().to(device)
        else:
            raise ValueError(f"Unsupported vfm_model: {vfm_model}")

    # ----------------------------------------------------------------------
    # VFM forward – returns ex_feats **and** attention map
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def vfm_forward(self, imgs_norm: torch.Tensor):
        """Compute *external features* (ex_feats) and, when available, the
        last‑block multi‑head attention map from the VFM.
        
        Parameters
        ----------
        imgs_norm : torch.Tensor
            Normalised images (B, 3, H, W) in the **same** resolution you feed
            to CLIP.
        Returns
        -------
        dict with keys:
            ex_feats : Tensor (B, C, H', W') – patch embeddings as spatial map
            attn     : Tensor (B, heads, N, N) – attention probabilities (only
                         when the backbone is ViT‑family; else None)
        """
        if self.vfm is None:
            return {'ex_feats': None, 'attn': None}

        vfm_dtype = next(self.vfm.parameters()).dtype
        imgs_norm = imgs_norm.to(dtype=vfm_dtype)

        if self.vfm_model.lower() in {'dino', 'dinov2'}:
            # --------------------------------------------------------------
            # ViT‑style backbones (DINO / DINOv2)
            # --------------------------------------------------------------
            feat_out = {}
            def _hook(module, _in, out):
                feat_out['qkv'] = out  # (B, N+1, 3*D)
            # register the hook only once
            if not hasattr(self, '_dino_hook_registered'):
                self.vfm.blocks[-1].attn.qkv.register_forward_hook(_hook)
                self._dino_hook_registered = True

            # Forward through VFM – we only need the *last* layer’s output
            feat = self.vfm.get_intermediate_layers(imgs_norm, n=1)[0]  # (B, N+1, D)

            B, N1, C = feat.shape  # N1 = N+1 (CLS + patches)
            if self.vfm_model.lower() == 'dino':
                N = N1 - 1             # number of patch tokens
            else: # dinov2
                N = N1
            H_feat = W_feat = int(math.sqrt(N))  # assumes square grid

            # 1. external features in (B, C, H', W') order
            if self.vfm_model.lower() == 'dino':    
                ex_feats = (
                    feat[:, 1:, :]                        # drop CLS
                    .reshape(B, H_feat, W_feat, C)        # (B, H', W', C)
                    .permute(0, 3, 1, 2).contiguous()     # (B, C, H', W')
                )
            else:
                ex_feats = (
                    feat.reshape(B, H_feat, W_feat, C)        # (B, H', W', C)
                    .permute(0, 3, 1, 2).contiguous()     # (B, C, H', W')
                )
            ex_feats = F.interpolate(ex_feats, size=(imgs_norm.shape[-2], imgs_norm.shape[-1]), mode='bilinear', align_corners=False)
            
            # Similarity attn map
            ex_feats = ex_feats.reshape(ex_feats.shape[0], ex_feats.shape[1], -1)
            ex_feats = F.normalize(ex_feats, dim=1)
            similarity = torch.einsum("b c m, b c n -> b m n", ex_feats, ex_feats)
            similarity[similarity < 0.3] = float('-inf')
            attn = F.softmax(similarity, dim=-1)

            return {'ex_feats': ex_feats, 'attn': attn} 

        elif self.vfm_model.lower() == 'sam':
            # SAM’s image encoder returns (B, C, H', W') directly
            patch_size = self.vfm.image_encoder.patch_embed.proj.kernel_size
            imgs_resized = F.interpolate(imgs_norm, size=(1024, 1024), mode='bilinear', align_corners=False)
            ex_feats = self.vfm.image_encoder(imgs_resized)  # already (B, C, H', W')
            return {'ex_feats': ex_feats, 'attn': None}

        elif self.vfm_model.lower() == 'mae':
            # MAE ViT backbone
            patch_size = self.vfm.patch_embed.patch_size
            imgs_resized = F.interpolate(imgs_norm, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            B = imgs_resized.size(0)
            I = imgs_resized.size(2) // patch_size[0]
            image_feat = self.vfm.forward_features(imgs_resized)        # (B, N, C)
            ex_feats = image_feat.view(B, I, I, -1).permute(0, 3, 1, 2)  # (B, C, I, I)
            return {'ex_feats': ex_feats, 'attn': None}

        else:
            # Should never reach here because of constructor validation
            return {'ex_feats': None, 'attn': None}

    @torch.no_grad()
    def viz_patch_attention(self, imgs, layer=-1, head=0, x_pix=None, y_pix=None, save_dir=None, dpi=150):
        idx_q = y_pix * imgs.shape[-2] + x_pix
        
        dir_path = os.path.dirname(save_dir)
        prefix   = os.path.basename(save_dir)

        attn = self.vfm_forward(imgs)['attn'] # (B, N, N)
        if attn is None:
            raise RuntimeError("Current VFM does not expose attention maps!")

        B, N, _ = attn.shape
        hw = int(math.sqrt(N))
        
        if idx_q is None:
            idx_q = (grid_h // 2) * grid_w + (grid_w // 2)  # centre patch

        img_disp = (imgs[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
        
        heat = attn[0][idx_q].cpu().reshape(hw, hw)  # (H', W')
        heat_up = F.interpolate(heat.unsqueeze(0).unsqueeze(0), size=imgs.shape[-2:], mode="bilinear", align_corners=False).squeeze(0).squeeze(0).detach().numpy()
        heat_up = heat_up / np.max(heat_up)

        # ── 1) idx_q → (cx, cy) 픽셀 좌표 ─────────────────────────────
        patch_h = imgs.shape[-2] // hw     # = patch_w
        patch_w = imgs.shape[-1] // hw
        row_q   = idx_q // hw
        col_q   = idx_q %  hw
        cy      = row_q * patch_h + patch_h // 2   # y (row)  중심
        cx      = col_q * patch_w + patch_w // 2   # x (col)  중심

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(
            1, 2, figsize=((imgs.shape[-1]*2)/dpi, imgs.shape[-2]/dpi), dpi=dpi)

        # 왼쪽 : 원본 + 표시점
        ax[0].imshow(img_disp)
        ax[0].scatter(cx, cy, s=8, c='red', marker='x', linewidths=1)
        ax[0].set_title(f"Query patch ({y_pix}, {x_pix})", fontsize=5)
        ax[0].axis("off")

        # 오른쪽 : 히트맵 overlay
        ax[1].imshow(img_disp)
        ax[1].imshow(heat_up, alpha=0.7, cmap="jet", vmin=0, vmax=1)
        ax[1].set_title("Attention heat‑map", fontsize=5)
        ax[1].axis("off")

        # ── 3) 저장 ────────────────────────────────────────────────
        file_name = f"{prefix}_{y_pix}_{x_pix}.png"
        file_path = os.path.join(dir_path, file_name)
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"Saved attention maps to '{file_path}'")


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
                v_img.half(), self.model_type,
                self.ignore_residual, output_cls_token=True)  # [1, D]

        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        cls_logits = cls_emb @ self.query_features.T          # [1, Q]
        return cls_logits

    def forward_feature(self, img, logit_size=None):
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
            
        # sigma = 4.0
        # blur_img = T.GaussianBlur(kernel_size=int(max(3, 4 * sigma + 1)), sigma=sigma)(img.float()).half()
        # blur_img_features = self.net.encode_image(blur_img, self.model_type, self.ignore_residual, self.output_cls_token)
        
        if self.output_cls_token:
            image_cls_token, image_features = image_features
            image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
            # cls_logits = image_cls_token @ self.query_features.T
            cls_logits = self.compute_cls_logits(img)
            # blur_img_cls_token, blur_img_features = blur_img_features

        # featup
        if self.feature_up:
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            # blur_img_features_up = blur_img_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            with torch.cuda.amp.autocast():
                image_features = self.upsampler(image_features, img).half()         
                # blur_img_features_up = self.upsampler(blur_img_features_up, blur_img).half()
            image_features = image_features.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)
            # blur_img_features_up = blur_img_features_up.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T

        # blur_img_features /= blur_img_features.norm(dim=-1, keepdim=True)
        # blur_img_features_up /= blur_img_features_up.norm(dim=-1, keepdim=True)
        # blur_logits = blur_img_features @ self.query_features.T
        # w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        # out_dim = blur_logits.shape[-1]
        # blur_logits = blur_logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        # blur_logits = nn.functional.interpolate(blur_logits, size=img.shape[-2:], mode='bilinear')
        # blur_logits = blur_logits.view(1, out_dim, image_w*image_h).permute(0, 2, 1)
        # blur_logits_up = blur_img_features_up @ self.query_features.T
        # logits = 3 * logits - 2 * (blur_logits_up - blur_logits)

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
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
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