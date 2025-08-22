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

    def ref_feature(self, img, logit_size=None):
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

        image_features = self.net.encode_last_layer(img)
        return ex_feats, image_features

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

    def ref_feature_clip(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]

        image_features = self.net.encode_last_layer(img)
        return image_features

    def _compute_miou(self, pred_mask, gt_mask, num_classes):
        """두 개의 분할 마스크 간의 mIoU를 계산합니다."""
        iou_list = []
        # pred_mask와 gt_mask가 Long 타입인지 확인
        pred_mask = pred_mask.long()
        gt_mask = gt_mask.long()

        for cls_idx in range(num_classes):
            pred_inds = (pred_mask == cls_idx)
            target_inds = (gt_mask == cls_idx)
            intersection = (pred_inds[target_inds]).long().sum().item()
            union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
            if union == 0:
                # union이 0인 경우 (둘 다 해당 클래스가 없는 경우) IoU는 1로 처리
                iou_list.append(1.0)
            else:
                iou_list.append(intersection / union)
        return np.mean(iou_list)

    def _predict_patch(self, patch_img):
        """단일 패치에 대한 분할 예측을 반환하는 헬퍼 함수"""
        # 외부 정보(ref) 없이 순수하게 패치 자체 정보로만 예측
        seg_logits = self.forward_feature(patch_img, ref_dino=None, ref_clip=None)
        
        # postprocess_result 로직을 가져와 argmax 예측 마스크 생성
        seg_logits = seg_logits[0] * self.logit_scale
        seg_logits = seg_logits.softmax(0)

        num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
        if num_cls != num_queries:
            seg_logits = seg_logits.unsqueeze(0)
            cls_index = nn.functional.one_hot(self.query_idx).T.view(num_cls, num_queries, 1, 1)
            seg_logits = (seg_logits * cls_index).max(1)[0]
        
        seg_pred = seg_logits.argmax(0, keepdim=True)
        seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx
        
        return seg_pred # [1, H, W] 형태의 예측 마스크

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


    def _get_upscaled_global_pred_map(self, full_img):
        """
        [추가된 헬퍼 함수]
        전체 이미지를 리사이즈 -> 예측 -> 다시 업스케일하여
        저해상도 컨텍스트를 반영한 예측 피처맵을 생성합니다.
        """
        h_img, w_img = full_img.shape[-2:]
        h_crop, w_crop = self.slide_crop, self.slide_crop # 224

        resized_full_img = F.interpolate(full_img, size=(h_crop, w_crop), mode='bilinear', align_corners=False)

        crop_dino, crop_clip = self.ref_feature(resized_full_img)
        image_features = self.net.encode_image(resized_full_img, self.model_type, self.ignore_residual, output_cls_token=False, ex_feats=crop_dino, ref_dino=None, ref_clip=None)
        
        # 리사이즈된 이미지의 DINO 피처맵 크기
        feature_h_resized = h_crop // self.dino_patch_size
        feature_w_resized = w_crop // self.dino_patch_size
        # 원본 이미지의 DINO 피처맵 크기
        target_feat_h = h_img // self.dino_patch_size
        target_feat_w = w_img // self.dino_patch_size

        image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_h_resized, feature_w_resized)
        image_features = F.interpolate(image_features, size=(target_feat_h, target_feat_w),
                                   mode='bilinear', align_corners=False)
        image_features = image_features.view(1, self.feat_dim, target_feat_w * target_feat_h).permute(0, 2, 1)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T # [1, 784, n_query]
        out_dim = logits.shape[-1]
        seg_logits = logits.permute(0, 2, 1).reshape(-1, out_dim, target_feat_h, target_feat_w)

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

        return seg_pred.squeeze(0) # [56, 56]

    def _get_ensembled_features(self, crop_img):
        """
        [추가된 헬퍼 함수]
        이미지를 0, 90, 180, 270도 회전시켜 피처를 각각 추출한 후,
        이를 평균내어 '앙상블된' DINO와 CLIP 피처를 반환합니다.
        """
        # 0도 (원본)
        dino_feat_0, clip_feat_0 = self.ref_feature(crop_img)

        # 90도
        img_90 = TF.rotate(crop_img, 90)
        dino_feat_90, clip_feat_90 = self.ref_feature(img_90)
        dino_feat_90 = torch.rot90(dino_feat_90, k=-1, dims=(2, 3))
        clip_feat_90 = torch.rot90(clip_feat_90, k=-1, dims=(2, 3))
        
        # 180도
        img_180 = TF.rotate(crop_img, 180)
        dino_feat_180, clip_feat_180 = self.ref_feature(img_180)
        dino_feat_180 = torch.rot90(dino_feat_180, k=-2, dims=(2, 3))
        clip_feat_180 = torch.rot90(clip_feat_180, k=-2, dims=(2, 3))
        
        # 270도
        img_270 = TF.rotate(crop_img, 270)
        dino_feat_270, clip_feat_270 = self.ref_feature(img_270)
        dino_feat_270 = torch.rot90(dino_feat_270, k=-3, dims=(2, 3))
        clip_feat_270 = torch.rot90(clip_feat_270, k=-3, dims=(2, 3))

        # 4개 피처를 평균내어 앙상블
        ensembled_dino_feat = torch.stack([dino_feat_0, dino_feat_90, dino_feat_180, dino_feat_270], dim=0).mean(dim=0)
        ensembled_clip_feat = torch.stack([clip_feat_0, clip_feat_90, clip_feat_180, clip_feat_270], dim=0).mean(dim=0)
        
        return ensembled_dino_feat, ensembled_clip_feat

    # Best
    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """
        [수정됨] 각 슬라이딩 패치에 대해 '자신을 제외한' 외부 정보로 클러스터링 및 샘플링 수행
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

                pred_map_0 = self._predict_feature_map(padded_img, x, dino_feat)

                if any(pad):
                    H_feat_orig = H_orig // self.dino_patch_size
                    W_feat_orig = W_orig // self.dino_patch_size
                    l_feat, _, t_feat, _ = [p // self.dino_patch_size for p in pad]
            
                    seg_pred = seg_pred[t_feat:t_feat + H_feat_orig, l_feat:l_feat + W_feat_orig]

                H_feat, W_feat = pred_map_0.shape
                total_feature_patches += H_feat * W_feat

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
                    kmeans = KMeans(n_clusters=K, init="k-means++", n_init=1, max_iter=25, random_state=42)
                    labels = torch.tensor(kmeans.fit_predict(external_dino_feats.cpu().numpy()), device=device, dtype=torch.long)
                    
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

                # crop_seg_logit = self.forward_feature(crop_img, ref_dino, ref_clip)

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