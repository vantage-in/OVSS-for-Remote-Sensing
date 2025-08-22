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
from sklearn.cluster import KMeans
import torchvision.transforms.functional as TF

@MODELS.register_module()
class ProxySegEarthSegmentationCat(BaseSegmentor):
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

    def forward_feature(self, img, ref_dino, ref_clip, logit_size=None, ex_feats=None):
        
        if ex_feats is None:
            ex_feats = self.ref_feature_dino(img, only_dino=True)
        
        if type(img) == list:
            img = img[0]
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
        else:
            image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, output_cls_token=False, ex_feats=ex_feats, ref_dino=ref_dino, ref_clip=ref_clip) #self.output_cls_token
        
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
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_h, feature_w)
            with torch.cuda.amp.autocast():
                # --- Upsample with refinement ---
                # image_features = self.upsampler.up2(image_features, img).half() # [1, 512, 28, 28]

                image_features = self.upsampler.up4(image_features, img).half() # [1, 512, 56, 56]]

                image_features = self.upsampler.up8(image_features, img).half() # [1, 512, 112, 112]

                image_features = self.upsampler.up16(image_features, img).half() # [1, 512, 224, 224]
        
                image_features = self.upsampler.fixup(image_features).half()
                # -------------------------------- 
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

        return logits

    def ref_feature(self, img, logit_size=None, only_dino=False):
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
        
        if type(img) == list:
            img = img[0]

        if only_dino:
            return ex_feats
        else:    
            image_features = self.net.encode_last_layer(img)
            return ex_feats, image_features

    def _predict_feature_map(self, padded_img, crop_dino, ori_shape, pad):
        """단일 패치에 대한 '피처맵' 단위의 분할 예측을 반환하는 헬퍼 함수"""
        image_features = self.net.encode_image(padded_img, self.model_type, self.ignore_residual, output_cls_token=False, ex_feats=crop_dino, ref_dino=None, ref_clip=None)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T # [1, 784, n_query]
        out_dim = logits.shape[-1]

        H_pad, W_pad = padded_img.shape[-2:]
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
        
        if any(pad):
            H_orig, W_orig = ori_shape
            H_feat_orig = H_orig // self.dino_patch_size
            W_feat_orig = W_orig // self.dino_patch_size
            l_feat, _, t_feat, _ = [p // self.dino_patch_size for p in pad]
            
            seg_pred = seg_pred[:, t_feat:t_feat + H_feat_orig, l_feat:l_feat + W_feat_orig]

        return seg_pred.squeeze(0) # [28, 28]

    # Fast
    def _compute_patch_data(self, crop_img, coords, patch_id):
        """
        crop_img -> 이미 pad된 상태, ori_shape, pad는 unpad용
        """
        patch_data = {'id': patch_id, 'coords': coords}

        dino_feats = {}


        H_orig, W_orig = crop_img.shape[-2:]
        patch_data['orig_shape'] = (H_orig, W_orig)
        pad_0 = self.compute_padsize(H_orig, W_orig, self.patch_size[0])
        if any(pad_0):
            padded_img_0 = F.pad(crop_img, pad_0, mode='constant', value=0) if any(pad_0) else crop_img
        else:
            padded_img_0 = crop_img
        dino_feats['0'], clip_0 = self.ref_feature(padded_img_0)

        img_90 = TF.rotate(crop_img, 90)
        H_90, W_90 = img_90.shape[-2:]
        pad_90 = self.compute_padsize(H_90, W_90, self.patch_size[0])
        if any(pad_90):
            padded_img_90 = F.pad(img_90, pad_90, mode='constant', value=0) if any(pad_90) else img_90
        else:
            padded_img_90 = img_90
        dino_feats['90'], clip_90 = self.ref_feature(padded_img_90)

        img_180 = TF.rotate(crop_img, 180)
        pad_180 = self.compute_padsize(H_orig, W_orig, self.patch_size[0])
        if any(pad_180):
            padded_img_180 = F.pad(img_180, pad_180, mode='constant', value=0) if any(pad_180) else img_180
        else:
            padded_img_180 = img_180
        dino_feats['180'], clip_180 = self.ref_feature(padded_img_180)

        img_270 = TF.rotate(crop_img, 270)
        pad_270 = self.compute_padsize(H_90, W_90, self.patch_size[0])
        if any(pad_270):
            padded_img_270 = F.pad(img_270, pad_270, mode='constant', value=0) if any(pad_270) else img_270
        else:
            padded_img_270 = img_270
        dino_feats['270'], clip_270 = self.ref_feature(padded_img_270)


        # --- 1. Reliability: Robustness check ---
        pred_map_0 = self._predict_feature_map(padded_img_0, dino_feats['0'], (H_orig, W_orig), pad_0)
        pred_map_90 = torch.rot90(self._predict_feature_map(padded_img_90, dino_feats['90'], (H_90, W_90), pad_90), k=-1, dims=(0, 1))
        pred_map_180 = torch.rot90(self._predict_feature_map(padded_img_180, dino_feats['180'], (H_orig, W_orig), pad_180), k=-2, dims=(0, 1))
        pred_map_270 = torch.rot90(self._predict_feature_map(padded_img_270, dino_feats['270'], (H_90, W_90), pad_270), k=-3, dims=(0, 1))
        
        patch_data['robustness_check_maps'] = {
            '0': pred_map_0, '90': pred_map_90, '180': pred_map_180, '270': pred_map_270
        }
        patch_data['feat_shape'] = pred_map_0.shape

        # --- 2. Ensemble features ---
        ensembled_dino = torch.stack([dino_feats['0'], dino_feats['90'], dino_feats['180'], dino_feats['270']], dim=0).mean(dim=0)
        ensembled_clip = torch.stack([clip_0, clip_90, clip_180, clip_270], dim=0).mean(dim=0)
        patch_data['ensembled_dino_feat'] = ensembled_dino
        patch_data['ensembled_clip_feat'] = ensembled_clip
        
        patch_data['dino_features'] = dino_feats
        patch_data['padded_images'] = {
            '0': padded_img_0, '90': padded_img_90, '180': padded_img_180, '270': padded_img_270
        }
        patch_data['pads'] = {'0': pad_0, '90': pad_90, '180': pad_180, '270': pad_270}

        return patch_data

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """
        [최적화됨] 모든 GPU 연산을 한 번의 루프로 통합하여 런타임 최소화
        """
        printing = False

        if type(img) == list:
            img = img[0].unsqueeze(0)
        
        h_stride, w_stride = (stride, stride)
        h_crop, w_crop = (crop_size, crop_size)
        batch_size, _, h_img, w_img = img.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        device = img.device

        # ========================================================================
        # Phase 1: 모든 패치에 대한 정보 일괄 계산 (Pre-computation)
        # ========================================================================
        if printing: print("Phase 1: Pre-computing all data for every patch...")
        
        patch_data_cache = []
        patch_counter = 0

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1, x1 = h_idx * h_stride, w_idx * w_stride
                y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
                y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                                
                # Compute in one-shot
                data = self._compute_patch_data(crop_img, (x1, y1, x2, y2), patch_counter)
                patch_data_cache.append(data)
                patch_counter += 1

        # ========================================================================
        # Phase 2: 강건한 앙상블 피처 풀 생성 (CPU-bound)
        # ========================================================================
        if printing: print("Phase 2: Building robust feature pool from cached data...")
        
        all_robust_dino, all_robust_clip, all_robust_patch_ids = [], [], []
        
        for data in patch_data_cache:
            maps = data['robustness_check_maps']
            robustness_mask = (maps['0'] == maps['90']) & (maps['0'] == maps['180']) & (maps['0'] == maps['270']) 
            
            if robustness_mask.sum() > 0:
                dino_flat = data['ensembled_dino_feat'].flatten(2, 3).permute(0, 2, 1).reshape(-1, data['ensembled_dino_feat'].shape[1])
                clip_flat = data['ensembled_clip_feat'].flatten(2, 3).permute(2, 0, 1).reshape(-1, data['ensembled_clip_feat'].shape[0] * data['ensembled_clip_feat'].shape[1])
                mask_flat = robustness_mask.flatten()

                robust_dino_feats_patch = dino_flat[mask_flat]
                all_robust_dino.append(robust_dino_feats_patch)
                all_robust_clip.append(clip_flat[mask_flat])
                all_robust_patch_ids.append(torch.full((robust_dino_feats_patch.shape[0],), data['id'], device=device))

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
        # Phase 3: 동적 샘플링 및 최종 예측 (GPU 연산 최소화)
        # ========================================================================
        if printing: print("Phase 3: Final prediction with dynamic sampling and TTA...")
        preds = img.new_zeros((batch_size, self.num_queries, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

        for data in patch_data_cache:
            # --- 동적 외부 정보 구성 ---
            ref_dino, ref_clip = None, None
            external_mask = (all_robust_patch_ids != data['id'])
            external_dino_feats = all_robust_dino_feats[external_mask]
            
            if external_dino_feats.shape[0] > 0:
                external_clip_feats = all_robust_clip_feats[external_mask]

                K = 25; M = 80
                K = min(K, external_dino_feats.shape[0])
                kmeans = KMeans(n_clusters=K, init="k-means++", n_init=1, max_iter=25, random_state=42)
                labels = torch.tensor(kmeans.fit_predict(external_dino_feats.cpu().numpy()), device=device, dtype=torch.long)
                dino_samples, clip_samples = [], []
                for cid in range(K):
                    member_indices = (labels == cid).nonzero(as_tuple=True)[0]
                    if len(member_indices) == 0: continue
                    
                    if len(member_indices) > M:
                        rand_indices = member_indices[torch.randperm(len(member_indices), device=device)[:M]]
                    else:
                        rand_indices = member_indices[torch.randint(0, len(member_indices), (M,), device=device)]
                    
                    dino_samples.append(external_dino_feats[rand_indices])
                    clip_samples.append(external_clip_feats[rand_indices])
                if dino_samples:
                    ref_dino = torch.cat(dino_samples, dim=0).t().unsqueeze(0).contiguous()
                    ref_clip = torch.cat(clip_samples, dim=0).view(-1, 12, 64).permute(1, 2, 0).contiguous()

            # --- 최종 예측 시 TTA 수행 ---
            padded_img_0 = data['padded_images']['0']
            dino_feat_0 = data['dino_features']['0']
            logit_0 = self.forward_feature(padded_img_0, ref_dino, ref_clip, ex_feats=dino_feat_0)
            
            padded_img_90 = data['padded_images']['90']
            dino_feat_90 = data['dino_features']['90']
            logit_90_raw = self.forward_feature(padded_img_90, ref_dino, ref_clip, ex_feats=dino_feat_90)
            logit_90 = torch.rot90(logit_90_raw, k=-1, dims=(2, 3)) # 결과 복원

            padded_img_180 = data['padded_images']['180']
            dino_feat_180 = data['dino_features']['180']
            logit_180_raw = self.forward_feature(padded_img_180, ref_dino, ref_clip, ex_feats=dino_feat_180)
            logit_180 = torch.rot90(logit_180_raw, k=-2, dims=(2, 3)) # 결과 복원

            padded_img_270 = data['padded_images']['270']
            dino_feat_270 = data['dino_features']['270']
            logit_270_raw = self.forward_feature(padded_img_270, ref_dino, ref_clip, ex_feats=dino_feat_270)
            logit_270 = torch.rot90(logit_270_raw, k=-3, dims=(2, 3)) # 결과 복원

            crop_seg_logit = torch.stack([logit_0, logit_90, logit_180, logit_270], dim=0).mean(dim=0)

            # --- 예측 결과에서 패딩 제거 ---
            pad = data['pads']['0']
            x1, y1, x2, y2 = data['coords']
            H_crop_orig, W_crop_orig = data['orig_shape']
            if any(pad):
                l, _, t, _ = pad
                # forward_feature의 출력은 패딩된 입력 크기(padded_crop_img)와 동일
                # 따라서 원본 crop 크기(H_crop_orig, W_crop_orig)만큼 잘라냄
                crop_seg_logit = crop_seg_logit[:, :, t:t + H_crop_orig, l:l + W_crop_orig]

            preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1

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