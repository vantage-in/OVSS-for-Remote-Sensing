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
            ex_feats = self.ref_feature_dino(img)
        
        if type(img) == list:
            img = img[0]
        if self.clip_type == 'BLIP':
            img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
            image_features = self.net.visual_encoder(img, self.ignore_residual)
            image_features = self.net.vision_proj(image_features[:, 1:, ])
        elif self.model_type == 'GEM':
            image_features = self.net.visual(img)
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
            handle = self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)

            # Forward pass in the model
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
            handle.remove()

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
            handle = self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)

            # Forward pass in the model
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
            handle.remove()

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

    # def forward_slide(self, img, img_metas, stride=112, crop_size=224):
    #     """Inference by sliding-window with overlap.
    #     If h_crop > h_img or w_crop > w_img, the small patch will be used to
    #     decode without padding.
    #     """
    #     if type(img) == list:
    #         img = img[0].unsqueeze(0)
    #     if type(stride) == int:
    #         stride = (stride, stride)
    #     if type(crop_size) == int:
    #         crop_size = (crop_size, crop_size)

    #     h_stride, w_stride = stride
    #     h_crop, w_crop = crop_size
    #     batch_size, _, h_img, w_img = img.shape
    #     out_channels = self.num_queries
    #     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    #     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    #     preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
    #     count_mat = img.new_zeros((batch_size, 1, h_img, w_img)) 

    #     # ---------- Reference img ------------------
    #     device = img.device
    #     K = 20  # number of global prototypes
    #     dino_list, clip_list = [], []
    #     patch_div_scores = []  # 슬라이딩 패치별 div score 저장
    #     patch_ids = []  # 각 feature가 속한 patch id 기록

    #     patch_counter = 0
    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             y1 = h_idx * h_stride
    #             x1 = w_idx * w_stride
    #             y2 = min(y1 + h_crop, h_img)
    #             x2 = min(x1 + w_crop, w_img)
    #             y1 = max(y2 - h_crop, 0)
    #             x1 = max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]

    #             # pad image when (image_size % patch_size != 0)
    #             H, W = crop_img.shape[2:]
    #             pad = self.compute_padsize(H, W, self.patch_size[0])

    #             if any(pad):
    #                 crop_img = nn.functional.pad(crop_img, pad)

    #             crop_dino, crop_clip = self.ref_feature(crop_img)
    #             # crop_dino: [1, 768, 28, 28]
    #             # crop_clip: [12, 64, 28, 28] 

    #             # 1) div_score
    #             # image_features = self.net.encode_image(img, self.model_type, self.ignore_residual, output_cls_token=False, ex_feats=crop_dino, ref_dino=None, ref_clip=None)
    #             # image_features /= image_features.norm(dim=-1, keepdim=True)
    #             # logits = image_features @ self.query_features.T # [1, 784, n_query]
    #             # out_dim = logits.shape[-1]
    #             # seg_logits = logits.permute(0, 2, 1).reshape(-1, out_dim, 28, 28)
    #             # seg_logits = seg_logits[0] * self.logit_scale
    #             # seg_logits = seg_logits.softmax(0)  # n_queries * w * h
    #             # num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
    #             # if num_cls != num_queries:
    #             #     seg_logits = seg_logits.unsqueeze(0)
    #             #     cls_index = nn.functional.one_hot(self.query_idx)
    #             #     cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
    #             #     seg_logits = (seg_logits * cls_index).max(1)[0]
    #             # seg_pred = seg_logits.argmax(0, keepdim=True)
    #             # seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx
    #             # seg_preds = seg_pred.flatten()
    #             # counts = torch.bincount(seg_preds, minlength=self.num_classes).float()
    #             # p = counts / counts.sum()
    #             # gini = 1.0 - torch.sum(p ** 2)
    #             # gini.item()
    #             # patch_div_scores.append(gini.item())

    #             # similarity = torch.einsum(
    #             #     "b c m, b c n -> b m n", 
    #             #     F.normalize(crop_dino.flatten(2, 3), dim=1), 
    #             #     F.normalize(crop_dino.flatten(2, 3), dim=1) 
    #             # ) # [1, 784, 784]
    #             # similarity[similarity < 0.0] = 0.0

    #             # token_norms = torch.norm(similarity, dim=-1)  # [1, 784]
    #             # norm_score = -token_norms.mean()              # norm 작을수록 점수 큼
    #             # patch_div_scores.append(norm_score.item())
                
    #             # 2) dino_feat
    #             crop_dino = F.normalize(crop_dino, dim=1)
    #             dino_list.append(crop_dino.flatten(2, 3).permute(0, 2, 1).reshape(-1, 768)) # [1, C, H, W] -> [H*W, C]

    #             # crop_clip: [n_head, head_dim, 28, 28]
    #             n_head, head_dim = crop_clip.shape[:2]
    #             # [n_head, head_dim, H, W] -> [H*W, n_head*head_dim]
    #             clip_list.append(
    #                 crop_clip.flatten(2, 3)                       # [n_head, head_dim, H*W]
    #                         .permute(2, 0, 1)                     # [H*W, n_head, head_dim]
    #                         .reshape(-1, n_head * head_dim)       # [H*W, 768]
    #             )

    #             patch_ids.extend([patch_counter] * (28*28))
    #             patch_counter += 1

    #     # best_patch_idx = int(np.argmax(patch_div_scores))
    #     # ref_dino = dino_list[best_patch_idx].t().unsqueeze(0).contiguous()    
    #     # ref_clip = clip_list[best_patch_idx].t().view(n_head, head_dim, 784)

    #     all_dino = torch.cat(dino_list, dim=0).to(dtype=torch.float32)  # [M, 768]
    #     all_clip = torch.cat(clip_list, dim=0).to(dtype=torch.float32)  # [M, 768]
    #     patch_ids = torch.tensor(patch_ids, device=device)
    #     B, C_d = all_dino.shape
    #     _, C_c = all_clip.shape
    #     K = min(K, B)

    #     # # ----- k-means++ 초기화 -----
    #     # def kmeans_plusplus_init(x, k):
    #     #     N = x.size(0)
    #     #     centroids = torch.empty(k, x.size(1), device=x.device, dtype=x.dtype)
    #     #     idx0 = torch.randint(0, N, (1,), device=x.device)
    #     #     centroids[0] = x[idx0]
    #     #     closest_dist2 = torch.cdist(x, centroids[0:1], p=2).squeeze(1).pow(2)
    #     #     for ci in range(1, k):
    #     #         probs = (closest_dist2 / (closest_dist2.sum() + 1e-12)).clamp(min=1e-12)
    #     #         next_idx = torch.multinomial(probs, 1)
    #     #         centroids[ci] = x[next_idx]
    #     #         dist2 = torch.cdist(x, centroids[ci:ci+1], p=2).squeeze(1).pow(2)
    #     #         closest_dist2 = torch.minimum(closest_dist2, dist2)
    #     #     return centroids

    #     # # ----- k-means 본체 -----
    #     # def kmeans_torch(x, k, iters=20):
    #     #     cent = kmeans_plusplus_init(x, k)
    #     #     labels = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
    #     #     for _ in range(iters):
    #     #         dists = torch.cdist(x, cent, p=2)
    #     #         new_labels = torch.argmin(dists, dim=1)

    #     #         if torch.equal(new_labels, labels):
    #     #             break
    #     #         labels = new_labels
    #     #         cent.zero_()
    #     #         counts = torch.bincount(labels, minlength=k).clamp(min=1).unsqueeze(1).to(x.dtype)
    #     #         cent.index_add_(0, labels, x)
    #     #         cent = cent / counts
    #     #         cent = F.normalize(cent, dim=1)
    #     #     return cent, labels

    #     # dino_centroids, labels = kmeans_torch(all_dino, K, iters=25)

    #     from sklearn.cluster import KMeans
    #     # ---- k-means (scikit-learn) -----
    #     def kmeans_sklearn(x, k, iters=25, seed=42):
    #         """
    #         x: torch.Tensor [N, C]
    #         return: centroids (torch.Tensor [k, C]), labels (torch.Tensor [N])
    #         """
    #         # numpy 변환
    #         x_np = x.detach().cpu().numpy()

    #         # scikit-learn KMeans
    #         kmeans = KMeans(
    #             n_clusters=k,
    #             init="k-means++",
    #             n_init=1,            # k-means++ 한 번 초기화
    #             max_iter=iters,
    #             random_state=seed,
    #             verbose=0
    #         )
    #         labels_np = kmeans.fit_predict(x_np)
    #         centroids_np = kmeans.cluster_centers_

    #         # torch 변환 
    #         centroids = torch.tensor(centroids_np, device=x.device, dtype=x.dtype)
    #         labels = torch.tensor(labels_np, device=x.device, dtype=torch.long)

    #         return centroids, labels

    #     dino_centroids, labels = kmeans_sklearn(all_dino, K, iters=25, seed=42)

    #     # # --- Cluster별 대표 CLIP 선택 ---
    #     # clip_proto = torch.zeros(K, n_head * head_dim, device=device)
    #     # for cid in range(K):
    #     #     member_idx = (labels == cid).nonzero(as_tuple=True)[0]
    #     #     if len(member_idx) == 0:
    #     #         continue
    #     #     # 해당 cluster 내에서 가장 다양성이 높은 patch id
    #     #     cluster_patch_ids = patch_ids[member_idx]
    #     #     best_patch = max(cluster_patch_ids.unique(),
    #     #                     key=lambda pid: patch_div_scores[pid])
    #     #     # best_patch에 속한 멤버 중 centroid에 가장 가까운 DINO feature 선택
    #     #     best_patch_idx = member_idx[cluster_patch_ids == best_patch]
    #     #     dino_sub = all_dino[best_patch_idx]
    #     #     dist = torch.norm(dino_sub - dino_centroids[cid], dim=1)
    #     #     chosen_idx = best_patch_idx[torch.argmin(dist)]
    #     #     clip_proto[cid] = all_clip[chosen_idx]

    #     # # [K, 768] -> [n_head, head_dim, 28, 28]
    #     # clip_proto = clip_proto.view(K, n_head, head_dim)   # [K, n_head, head_dim]
    #     # clip_proto = clip_proto.permute(1, 2, 0).contiguous()  # [n_head, head_dim, K]
        
    #     # # ----- [1, C, K] 형태로 저장 -----
    #     # ref_dino = dino_centroids.t().unsqueeze(0).contiguous()
    #     # ref_clip = clip_proto
    #     # # --- Cluster별 대표 CLIP 선택 ---



    #     # # --- Cluster별 대표 DINO 선택 (2가지 옵션 중 하나) ---
    #     # dino_select_mode = "exclude_highest"
    #     # dino_proto = torch.zeros(K, 768, device=device)

    #     # for cid in range(K):
    #     #     member_idx = (labels == cid).nonzero(as_tuple=True)[0]
    #     #     if len(member_idx) == 0:
    #     #         continue

    #     #     cluster_patch_ids = patch_ids[member_idx]

    #     #     if dino_select_mode == "lowest":  
    #     #         # 1) cluster 내에서 div score 가장 낮은 patch id
    #     #         worst_patch = min(cluster_patch_ids.unique(),
    #     #                         key=lambda pid: patch_div_scores[int(pid)])
    #     #         # 해당 patch의 모든 DINO feature 평균
    #     #         worst_patch_idx = member_idx[cluster_patch_ids == worst_patch]
    #     #         dino_proto[cid] = all_dino[worst_patch_idx].mean(dim=0)

    #     #     elif dino_select_mode == "exclude_highest":
    #     #         # 2) cluster 내에서 div score 가장 높은 patch id
    #     #         best_patch = max(cluster_patch_ids.unique(),
    #     #                         key=lambda pid: patch_div_scores[int(pid)])
    #     #         # 해당 patch를 제외한 나머지 feature 평균
    #     #         keep_idx = member_idx[cluster_patch_ids != best_patch]
    #     #         if len(keep_idx) > 0:
    #     #             dino_proto[cid] = all_dino[keep_idx].mean(dim=0)
    #     #         else:
    #     #             # 예외: 남는 feature가 없으면 centroid 사용
    #     #             dino_proto[cid] = dino_centroids[cid]

    #     # # [K, 768] → [1, 768, K]
    #     # ref_dino = dino_proto.t().unsqueeze(0).contiguous()
    #     # # --- Cluster별 대표 DINO 선택 (2가지 옵션 중 하나) ---



    #     # ------ Random Sampling -------
    #     M = 40  # cluster당 샘플 개수
    #     dino_samples = []
    #     clip_samples = []

    #     for cid in range(K):
    #         member_idx = (labels == cid).nonzero(as_tuple=True)[0]  # 해당 cluster의 index
    #         if len(member_idx) == 0:
    #             continue
            
    #         torch.manual_seed(42) 
    #         # cluster 내에서 M개 랜덤 샘플 (M > len(member_idx)면 모두 사용)
    #         if len(member_idx) > M:
    #             rand_idx = member_idx[torch.randperm(len(member_idx), device=device)[:M]]
    #         else:
    #             extra_needed = M - len(member_idx)
    #             dup_idx = member_idx[torch.randint(0, len(member_idx), (extra_needed,), device=device)]
    #             rand_idx = torch.cat([member_idx, dup_idx], dim=0)

    #         dino_samples.append(all_dino[rand_idx])  # [M', 768]
    #         clip_samples.append(all_clip[rand_idx])  # [M', clip_dim]

    #     # concat [K*M, ...]
    #     dino_samples = torch.cat(dino_samples, dim=0)  # [K*M, dino_dim] 
    #     clip_samples = torch.cat(clip_samples, dim=0)  # [K*M, clip_dim]

    #     ref_dino = dino_samples.t().unsqueeze(0).contiguous()
    #     clip_samples = clip_samples.view(K*M, n_head, head_dim)   # [K, n_head, head_dim]
    #     ref_clip = clip_samples.permute(1, 2, 0).contiguous()  # [n_head, head_dim, K]
    #     # ------ Random Sampling -------



    #     # # ----- Random sampling + ranking -----
    #     # M = 2
    #     # dino_samples = []
    #     # clip_samples = []

    #     # torch.manual_seed(42)  # reproducibility

    #     # for cid in range(K):
    #     #     member_idx = (labels == cid).nonzero(as_tuple=True)[0]
    #     #     if len(member_idx) == 0:
    #     #         continue

    #     #     # ---- CLIP 우선순위 선택 ----
    #     #     cluster_patch_ids = patch_ids[member_idx]
    #     #     dino_feats = all_dino[member_idx]
    #     #     clip_feats = all_clip[member_idx]

    #     #     # centroid
    #     #     centroid = dino_centroids[cid].unsqueeze(0)

    #     #     # patch 우선순위 (높은 점수 먼저)
    #     #     patch_scores = torch.tensor([patch_div_scores[int(pid)] for pid in cluster_patch_ids],
    #     #                                 device=device)
    #     #     dist_to_centroid = torch.norm(dino_feats - centroid, dim=1)

    #     #     # 1차 정렬: 거리 오름차순
    #     #     dist_sort_idx = torch.argsort(dist_to_centroid, dim=0)  
    #     #     # 2차 정렬: patch score 내림차순 (우선 적용)
    #     #     score_sort_idx = torch.argsort(patch_scores, descending=True, dim=0)

    #     #     # 두 기준을 결합: score 순서 우선 → 그 안에서 거리 순서 유지
    #     #     sort_idx = sorted(range(len(member_idx)), 
    #     #                     key=lambda i: (-patch_scores[i].item(), dist_to_centroid[i].item()))
    #     #     sort_idx = torch.tensor(sort_idx, device=device)

    #     #     if len(sort_idx) >= M:
    #     #         chosen_idx_clip = member_idx[sort_idx[:M]]
    #     #     else:
    #     #         # 중복 허용해서 M개 채우기
    #     #         reps = (M + len(sort_idx) - 1) // len(sort_idx)  # 필요한 반복 횟수
    #     #         repeated = sort_idx.repeat(reps)[:M]
    #     #         chosen_idx_clip = member_idx[repeated]
    #     #     clip_samples.append(all_clip[chosen_idx_clip])

    #     #     # ---- DINO 선택 ----
    #     #     chosen_idx_dino_list = []
    #     #     chosen_patches_clip = patch_ids[chosen_idx_clip]
    #     #     mask_diff_patch = ~torch.isin(cluster_patch_ids, chosen_patches_clip)
    #     #     diff_patch_idx = member_idx[mask_diff_patch]

    #     #     # 다른 patch에서 선택
    #     #     if len(diff_patch_idx) > 0:
    #     #         dino_diff_feats = all_dino[diff_patch_idx]
    #     #         for dino_ref in all_dino[chosen_idx_clip]:
    #     #             if len(dino_diff_feats) == 0:
    #     #                 break
    #     #             dist = torch.norm(dino_diff_feats - dino_ref.unsqueeze(0), dim=1)
    #     #             nearest_idx = diff_patch_idx[torch.argmin(dist)]
    #     #             chosen_idx_dino_list.append(int(nearest_idx))  # int로 변환해서 저장
    #     #             # 이미 선택된 건 제거
    #     #             keep_mask = diff_patch_idx != nearest_idx
    #     #             diff_patch_idx = diff_patch_idx[keep_mask]
    #     #             dino_diff_feats = all_dino[diff_patch_idx]

    #     #     # 부족하면 낮은 patch_div_scores 순으로 채움
    #     #     if len(chosen_idx_dino_list) < M:
    #     #         remaining_needed = M - len(chosen_idx_dino_list)
    #     #         remaining_idx = member_idx[~torch.isin(member_idx, torch.tensor(chosen_idx_dino_list, device=device))]
    #     #         remaining_patch_scores = torch.tensor(
    #     #             [patch_div_scores[int(pid)] for pid in patch_ids[remaining_idx]],
    #     #             device=device
    #     #         )
    #     #         sort_low_idx = torch.argsort(remaining_patch_scores)
    #     #         fill_idx = remaining_idx[sort_low_idx[:remaining_needed]]
    #     #         if len(fill_idx) < remaining_needed:
    #     #             # 중복 허용해서 M개 채우기
    #     #             reps = (remaining_needed + len(fill_idx) - 1) // max(len(fill_idx), 1)
    #     #             fill_idx = fill_idx.repeat(reps)[:remaining_needed]

    #     #         chosen_idx_dino_list.extend([int(idx) for idx in fill_idx])

    #     #     # 최종 tensor index로 변환
    #     #     chosen_idx_dino = torch.tensor(chosen_idx_dino_list, device=device, dtype=torch.long)

    #     #     # 안전하게 index
    #     #     dino_samples.append(all_dino[chosen_idx_dino])

    #     # # 합치기
    #     # dino_samples = torch.cat(dino_samples, dim=0)
    #     # clip_samples = torch.cat(clip_samples, dim=0)

    #     # ref_dino = dino_samples.t().unsqueeze(0).contiguous()
    #     # clip_samples = clip_samples.view(K*M, n_head, head_dim)   # [K, n_head, head_dim]
    #     # ref_clip = clip_samples.permute(1, 2, 0).contiguous()  # [n_head, head_dim, K]
    #     # # ----- Random sampling + ranking -----



    #     # ref_dino = None
    #     # ref_clip = None


    #     # y1 = (h_grids - 1) * h_stride
    #     # x1 = 0 * w_stride
    #     # y2 = min(y1 + h_crop, h_img)
    #     # x2 = min(x1 + w_crop, w_img)
    #     # y1 = max(y2 - h_crop, 0)
    #     # x1 = max(x2 - w_crop, 0)
    #     # ref_img = img[:, :, y1:y2, x1:x2]

    #     # ref_dino, ref_clip = self.ref_feature(ref_img)
    #     # ---------- Reference img ------------------

    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             y1 = h_idx * h_stride
    #             x1 = w_idx * w_stride
    #             y2 = min(y1 + h_crop, h_img)
    #             x2 = min(x1 + w_crop, w_img)
    #             y1 = max(y2 - h_crop, 0)
    #             x1 = max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]

    #             # pad image when (image_size % patch_size != 0)
    #             H, W = crop_img.shape[2:]
    #             pad = self.compute_padsize(H, W, self.patch_size[0])

    #             if any(pad):
    #                 crop_img = nn.functional.pad(crop_img, pad)
                
    #             crop_seg_logit = self.forward_feature(crop_img, ref_dino, ref_clip)

    #             # mask cutting for padded image
    #             if any(pad):
    #                 l, t = pad[0], pad[2]
    #                 crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

    #             preds += nn.functional.pad(crop_seg_logit,
    #                                        (int(x1), int(preds.shape[3] - x2), int(y1),
    #                                         int(preds.shape[2] - y2)))

    #             count_mat[:, :, y1:y2, x1:x2] += 1
    #     assert (count_mat == 0).sum() == 0

    #     preds = preds / count_mat
    #     img_size = img_metas[0]['ori_shape'][:2]
    #     logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

    #     return logits

    # def forward_slide(self, img, img_metas, stride=112, crop_size=224, consistency_threshold=0.5):
    #     """
    #     [수정됨] 정밀도 기반 선택적 프로토타이핑을 적용한 슬라이딩 윈도우 추론
    #     """
    #     if type(img) == list:
    #         img = img[0].unsqueeze(0)
        
    #     h_stride, w_stride = (stride, stride)
    #     h_crop, w_crop = (crop_size, crop_size)
    #     batch_size, _, h_img, w_img = img.shape
    #     out_channels = self.num_queries
    #     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    #     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        
    #     device = img.device
        
    #     # ========================================================================
    #     # Phase 1: 사전 평가 및 일관성 점수 계산 (Pre-evaluation & Scoring)
    #     # ========================================================================
    #     print("Phase 1: Calculating consistency scores for all patches...")
    #     all_patches_data = []
    #     patch_counter = 0

    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             y1 = h_idx * h_stride
    #             x1 = w_idx * w_stride
    #             y2 = min(y1 + h_crop, h_img)
    #             x2 = min(x1 + w_crop, w_img)
    #             y1 = max(y2 - h_crop, 0)
    #             x1 = max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]

    #             # --- 1. 변환 일관성(Augmentation Consistency) 점수 계산 ---
    #             # 원본 예측
    #             pred_original = self._predict_patch(crop_img)
                
    #             # 회전 변환 및 예측
    #             angles = [90, 180, 270]
    #             consistency_scores = []
    #             for angle in angles:
    #                 # 회전 및 예측
    #                 rotated_img = TF.rotate(crop_img, angle)
    #                 pred_rotated = self._predict_patch(rotated_img)
                    
    #                 # 예측 결과를 다시 원위치로 회전
    #                 pred_restored = TF.rotate(pred_rotated, -angle)
                    
    #                 # mIoU 계산
    #                 miou = self._compute_miou(pred_original, pred_restored, self.num_classes)
    #                 consistency_scores.append(miou)
                
    #             # 평균 mIoU를 최종 일관성 점수로 사용
    #             consistency_score = np.mean(consistency_scores)

    #             # --- 2. 피처 추출 ---
    #             # NOTE: 공간적 일관성은 구현이 복잡하므로 여기서는 변환 일관성만 사용합니다.
    #             # 필요 시 이 부분에 인접 패치와의 mIoU 비교 로직을 추가할 수 있습니다.
    #             crop_dino, crop_clip = self.ref_feature(crop_img)

    #             all_patches_data.append({
    #                 'id': patch_counter,
    #                 'coords': (x1, y1, x2, y2),
    #                 'dino_feat': crop_dino.flatten(2, 3).permute(0, 2, 1).reshape(-1, 768),
    #                 'clip_feat': crop_clip.flatten(2, 3).permute(2, 0, 1).reshape(-1, crop_clip.shape[0] * crop_clip.shape[1]),
    #                 'consistency': consistency_score,
    #                 'patch_img': crop_img
    #             })
    #             patch_counter += 1

    #     # ========================================================================
    #     # Phase 2: 선택적 프로토타이핑 (Selective Prototyping)
    #     # ========================================================================
    #     print(f"Phase 2: Filtering patches with consistency >= {consistency_threshold} and sampling prototypes...")
        
    #     # --- 1. 일관성 점수 기반 필터링 ---
    #     reliable_patches = [p for p in all_patches_data if p['consistency'] >= consistency_threshold]
    #     if not reliable_patches:
    #         print("Warning: No reliable patches found. Using all patches for prototyping.")
    #         reliable_patches = all_patches_data

    #     reliable_patch_indices = [p['id'] for p in reliable_patches]
    #     print(f"  - Found {len(reliable_patches)} / {len(all_patches_data)} reliable patches.")
    #     print(f"  - Indices of reliable patches: {reliable_patch_indices}")

    #     reliable_dino_feats = torch.cat([p['dino_feat'] for p in reliable_patches], dim=0).to(dtype=torch.float32)
    #     reliable_clip_feats = torch.cat([p['clip_feat'] for p in reliable_patches], dim=0).to(dtype=torch.float32)

    #     # --- 2. K-means 클러스터링 (신뢰성 있는 피처 대상) ---
    #     K = 20 # 프로토타입 개수
    #     B, C_d = reliable_dino_feats.shape
    #     K = min(K, B)

    #     kmeans = KMeans(n_clusters=K, init="k-means++", n_init=1, max_iter=25, random_state=42)
    #     labels = torch.tensor(kmeans.fit_predict(reliable_dino_feats.cpu().numpy()), device=device, dtype=torch.long)
        
    #     # --- 3. 클러스터 내 랜덤 샘플링 (신뢰성 있는 피처 대상) ---
    #     M = 40  # 클러스터 당 샘플 개수
    #     dino_samples, clip_samples = [], []

    #     for cid in range(K):
    #         member_indices = (labels == cid).nonzero(as_tuple=True)[0]
    #         if len(member_indices) == 0:
    #             continue
            
    #         # 클러스터 내에서 M개 랜덤 샘플
    #         if len(member_indices) > M:
    #             rand_indices = member_indices[torch.randperm(len(member_indices), device=device)[:M]]
    #         else: # 부족하면 중복 허용하여 샘플링
    #             rand_indices = member_indices[torch.randint(0, len(member_indices), (M,), device=device)]

    #         dino_samples.append(reliable_dino_feats[rand_indices])
    #         clip_samples.append(reliable_clip_feats[rand_indices])
        
    #     # 최종 프로토타입(외부 정보) 생성
    #     ref_dino = torch.cat(dino_samples, dim=0).t().unsqueeze(0).contiguous()
    #     # n_head, head_dim = self.net.encode_last_layer(img).shape[:2] # clip head 정보 가져오기
    #     n_head = 12
    #     head_dim = 64
    #     ref_clip = torch.cat(clip_samples, dim=0).view(-1, n_head, head_dim).permute(1, 2, 0).contiguous()
        
    #     # ========================================================================
    #     # Phase 3: 최종 예측 (Final Prediction)
    #     # ========================================================================
    #     print("Phase 3: Performing final prediction using selective prototypes...")
    #     preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
    #     count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

    #     for patch_data in all_patches_data:
    #         x1, y1, x2, y2 = patch_data['coords']
    #         crop_img = patch_data['patch_img']

    #         # 선택된 프로토타입을 이용해 최종 예측 수행
    #         crop_seg_logit = self.forward_feature(crop_img, ref_dino, ref_clip)

    #         preds += nn.functional.pad(crop_seg_logit,
    #                                    (int(x1), int(preds.shape[3] - x2), int(y1),
    #                                     int(preds.shape[2] - y2)))
    #         count_mat[:, :, y1:y2, x1:x2] += 1
            
    #     assert (count_mat == 0).sum() == 0
    #     preds = preds / count_mat
    #     img_size = img_metas[0]['ori_shape'][:2]
    #     logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

    #     return logits

    # def _predict_feature_map(self, padded_img, crop_dino, ori_shape, pad):
    #     """단일 패치에 대한 '피처맵' 단위의 분할 예측을 반환하는 헬퍼 함수"""
    #     image_features = self.net.encode_image(padded_img, self.model_type, self.ignore_residual, output_cls_token=False, ex_feats=crop_dino, ref_dino=None, ref_clip=None)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #     logits = image_features @ self.query_features.T # [1, 784, n_query]
    #     out_dim = logits.shape[-1]

    #     H_pad, W_pad = padded_img.shape[-2:]
    #     H_feat_pad, W_feat_pad = H_pad // self.dino_patch_size, W_pad // self.dino_patch_size

    #     seg_logits = logits.permute(0, 2, 1).reshape(-1, out_dim, H_feat_pad, W_feat_pad)
    #     seg_logits = seg_logits[0] * self.logit_scale
    #     seg_logits = seg_logits.softmax(0)  # n_queries * w * h
    #     num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
    #     if num_cls != num_queries:
    #         seg_logits = seg_logits.unsqueeze(0)
    #         cls_index = nn.functional.one_hot(self.query_idx)
    #         cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
    #         seg_logits = (seg_logits * cls_index).max(1)[0]
    #     seg_pred = seg_logits.argmax(0, keepdim=True)
    #     seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx
        
    #     if any(pad):
    #         H_orig, W_orig = ori_shape
    #         H_feat_orig = H_orig // self.dino_patch_size
    #         W_feat_orig = W_orig // self.dino_patch_size
    #         l_feat, _, t_feat, _ = [p // self.dino_patch_size for p in pad]
            
    #         seg_pred = seg_pred[:, t_feat:t_feat + H_feat_orig, l_feat:l_feat + W_feat_orig]

    #     return seg_pred.squeeze(0) # [28, 28]

    def _predict_feature_map(self, patch_img):
        """단일 패치에 대한 '피처맵' 단위의 분할 예측을 반환하는 헬퍼 함수"""
        H_orig, W_orig = patch_img.shape[-2:]
        pad = self.compute_padsize(H_orig, W_orig, self.patch_size[0])

        if any(pad):
            padded_img = F.pad(patch_img, pad, mode='constant', value=0)
        else:
            padded_img = patch_img

        crop_dino = self.ref_feature_dino(padded_img)

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
            H_feat_orig = H_orig // self.dino_patch_size
            W_feat_orig = W_orig // self.dino_patch_size
            l_feat, _, t_feat, _ = [p // self.dino_patch_size for p in pad]
            
            seg_pred = seg_pred[:, t_feat:t_feat + H_feat_orig, l_feat:l_feat + W_feat_orig]

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

    # def forward_slide(self, img, img_metas, stride=112, crop_size=224):
    #     """
    #     [수정됨] 개별 피처 패치 단위의 강건성 기반 선택적 프로토타이핑
    #     """
    #     printing = True

    #     if type(img) == list:
    #         img = img[0].unsqueeze(0)
        
    #     h_stride, w_stride = (stride, stride)
    #     h_crop, w_crop = (crop_size, crop_size)
    #     batch_size, _, h_img, w_img = img.shape
    #     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    #     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    #     device = img.device
        
    #     # ========================================================================
    #     # Phase 1: 강건한(Robust) 개별 피처 임베딩 수집
    #     # ========================================================================
    #     if printing: print("Phase 1: Collecting robust feature embeddings based on rotation consistency...") 
        
    #     robust_dino_list = []
    #     robust_clip_list = []
    #     total_feature_patches = 0

    #     # multi-scale
    #     pred_map_global_scaled = self._get_upscaled_global_pred_map(img)

    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             # 1. 슬라이딩 패치 추출
    #             y1, x1 = h_idx * h_stride, w_idx * w_stride
    #             y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
    #             y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]

    #             # 2. 원본 및 회전된 패치에 대한 예측 피처맵 생성
    #             # 원본 예측
    #             pred_map_0 = self._predict_feature_map(crop_img)
    #             H_feat, W_feat = pred_map_0.shape
    #             total_feature_patches += H_feat * W_feat

    #             # 회전 예측 및 원위치 복원
    #             pred_map_90 = torch.rot90(self._predict_feature_map(TF.rotate(crop_img, 90)), k=-1, dims=(0, 1))
    #             pred_map_180 = torch.rot90(self._predict_feature_map(TF.rotate(crop_img, 180)), k=-2, dims=(0, 1))
    #             pred_map_270 = torch.rot90(self._predict_feature_map(TF.rotate(crop_img, 270)), k=-3, dims=(0, 1))

    #             y1_feat = y1 // self.dino_patch_size
    #             x1_feat = x1 // self.dino_patch_size
    #             # crop된 피처맵의 실제 크기(H_feat, W_feat)를 사용
    #             y2_feat = y1_feat + H_feat
    #             x2_feat = x1_feat + W_feat
    #             pred_map_scale_crop = pred_map_global_scaled[y1_feat:y2_feat, x1_feat:x2_feat]

    #             # 3. 강건성 마스크 생성
    #             # 4개의 예측이 모두 동일한 위치를 True로 하는 boolean 마스크 생성
    #             robustness_mask = (pred_map_0 == pred_map_90) & \
    #                               (pred_map_0 == pred_map_180) & \
    #                               (pred_map_0 == pred_map_270) & \
    #                               (pred_map_0 == pred_map_scale_crop)
                                  
    #             # 4. 강건한 임베딩만 필터링하여 수집
    #             if robustness_mask.sum() > 0: # 강건한 패치가 하나라도 있다면
    #                 # 원본 패치에서 DINO/CLIP 피처 추출
    #                 dino_feat, clip_feat = self.ref_feature(crop_img)
                    
    #                 # 피처를 [H*W, C] 형태로 변환
    #                 dino_feat_flat = dino_feat.flatten(2, 3).permute(0, 2, 1).reshape(-1, dino_feat.shape[1])
    #                 clip_feat_flat = clip_feat.flatten(2, 3).permute(2, 0, 1).reshape(-1, clip_feat.shape[0] * clip_feat.shape[1])
                    
    #                 # 마스크를 1D로 펴서 필터링
    #                 mask_flat = robustness_mask.flatten()
                    
    #                 robust_dino_list.append(dino_feat_flat[mask_flat])
    #                 robust_clip_list.append(clip_feat_flat[mask_flat])
        
    #     # 수집된 강건한 임베딩들을 하나로 합침
    #     robust_dino_feats = torch.cat(robust_dino_list, dim=0).to(dtype=torch.float32)
    #     robust_clip_feats = torch.cat(robust_clip_list, dim=0).to(dtype=torch.float32)
        
    #     num_robust_patches = robust_dino_feats.shape[0]
    #     if printing: print(f"  - Collected {num_robust_patches} / {total_feature_patches} robust feature embeddings.") 

    #     # ========================================================================
    #     # Phase 2: 선택적 프로토타이핑
    #     # ========================================================================
    #     if printing: print("Phase 2: Performing K-Means and sampling on robust embeddings...")

    #     if num_robust_patches == 0:
    #         print("Warning: No robust features found. Final prediction will be made without external context.")
    #         ref_dino, ref_clip = None, None
    #     else:
    #         # K-means Clutering
    #         K = 25
    #         K = min(K, num_robust_patches)
    #         kmeans = KMeans(n_clusters=K, init="k-means++", n_init=1, max_iter=25, random_state=42)
    #         labels = torch.tensor(kmeans.fit_predict(robust_dino_feats.cpu().numpy()), device=device, dtype=torch.long)
            
    #         # Random Sampling in each cluster
    #         M = 80
    #         dino_samples, clip_samples = [], []
    #         for cid in range(K):
    #             member_indices = (labels == cid).nonzero(as_tuple=True)[0]
    #             if len(member_indices) == 0: continue
                
    #             if len(member_indices) > M:
    #                 rand_indices = member_indices[torch.randperm(len(member_indices), device=device)[:M]]
    #             else:
    #                 rand_indices = member_indices[torch.randint(0, len(member_indices), (M,), device=device)]
                
    #             dino_samples.append(robust_dino_feats[rand_indices])
    #             clip_samples.append(robust_clip_feats[rand_indices])
            
    #         # 최종 프로토타입 생성
    #         ref_dino = torch.cat(dino_samples, dim=0).t().unsqueeze(0).contiguous()
    #         ref_clip = torch.cat(clip_samples, dim=0).view(-1, 12, 64).permute(1, 2, 0).contiguous()

    #     # ========================================================================
    #     # Phase 3: 최종 예측
    #     # ========================================================================
    #     if printing: print("Phase 3: Performing final prediction using robust prototypes...") 
    #     preds = img.new_zeros((batch_size, self.num_queries, h_img, w_img))
    #     count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             y1, x1 = h_idx * h_stride, w_idx * w_stride
    #             y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
    #             y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]

    #             crop_seg_logit = self.forward_feature(crop_img, ref_dino, ref_clip)

    #             preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
    #             count_mat[:, :, y1:y2, x1:x2] += 1
            
    #     assert (count_mat == 0).sum() == 0
    #     preds = preds / count_mat
    #     img_size = img_metas[0]['ori_shape'][:2]
    #     logits = F.interpolate(preds, size=img_size, mode='bilinear')

    #     return logits

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

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # ... (crop_img 생성 및 강건성 마스크 계산 로직은 동일) ...
                y1, x1 = h_idx * h_stride, w_idx * w_stride
                y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
                y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                dino_feat, clip_feat = self.ref_feature(crop_img)
                dino_feat_flat = dino_feat.flatten(2, 3).permute(0, 2, 1).reshape(-1, dino_feat.shape[1])
                clip_feat_flat = clip_feat.flatten(2, 3).permute(2, 0, 1).reshape(-1, clip_feat.shape[0] * clip_feat.shape[1])

                all_robust_dino.append(dino_feat_flat)
                all_robust_clip.append(clip_feat_flat)

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

                crop_seg_logit = self.forward_feature(padded_crop_img, ref_dino, ref_clip)

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

    # Fast
    # def _compute_patch_data(self, crop_img, coords, patch_id, ori_shape, pad):
    #     """
    #     crop_img -> 이미 pad된 상태, ori_shape, pad는 unpad용
    #     """
    #     patch_data = {'id': patch_id, 'coords': coords}

    #     dino_feats = {}



    #     H_orig, W_orig = crop_img.shape[-2:]
    #     pad_0 = self.compute_padsize(H_orig, W_orig, self.patch_size[0])
    #     if any(pad_0):
    #         padded_img_0 = F.pad(crop_img, pad_0, mode='constant', value=0) if any(pad_0) else crop_img
    #     else:
    #         padded_img_0 = crop_img
    #     dino_feats['0'] = self.ref_feature_dino(padded_img_0)

    #     img_90 = TF.rotate(crop_img, 90)
    #     H_90, W_90 = img_90.shape[-2:]
    #     pad_90 = self.compute_padsize(H_90, W_90, self.patch_size[0])
    #     if any(pad_90):
    #         padded_img_90 = F.pad(img_90, pad_90, mode='constant', value=0) if any(pad_90) else img_90
    #     else:
    #         padded_img_90 = img_09
    #     dino_feats['90'] = self.ref_feature_dino(padded_img_90)

    #     img_180 = TF.rotate(crop_img, 180)
    #     pad_180 = self.compute_padsize(H_orig, W_orig, self.patch_size[0])
    #     if any(pad_180):
    #         padded_img_180 = F.pad(img_180, pad_180, mode='constant', value=0) if any(pad_180) else img_180
    #     else:
    #         padded_img_180 = img_180
    #     dino_feats['180'] = self.ref_feature_dino(padded_img_180)

    #     img_270 = TF.rotate(crop_img, 270)
    #     pad_270 = self.compute_padsize(H_90, W_90, self.patch_size[0])
    #     if any(pad_270):
    #         padded_img_270 = F.pad(img_270, pad_270, mode='constant', value=0) if any(pad_270) else img_270
    #     else:
    #         padded_img_270 = img_270
    #     dino_feats['270'] = self.ref_feature_dino(padded_img_270)


    #     # --- 1. Reliability: Robustness check ---
    #     pred_map_0 = self._predict_feature_map(crop_img, dino_feats['0'], (H_orig, W_orig), pad_0)
    #     H_feat, W_feat = pred_map_0.shape

    #     pred_map_90 = torch.rot90(self._predict_feature_map(TF.rotate(crop_img, 90), dino_90, ), k=-1, dims=(0, 1)) # ori shape, pad 어캐 처리??
    #     pred_map_180 = torch.rot90(self._predict_feature_map(TF.rotate(crop_img, 180)), k=-2, dims=(0, 1))
    #     pred_map_270 = torch.rot90(self._predict_feature_map(TF.rotate(crop_img, 270)), k=-3, dims=(0, 1))
        
    #     y1_feat, x1_feat = coords[1] // self.dino_patch_size, coords[0] // self.dino_patch_size
    #     y2_feat, x2_feat = y1_feat + H_feat, x1_feat + W_feat
        
    #     patch_data['robustness_check_maps'] = {
    #         '0': pred_map_0, '90': pred_map_90, '180': pred_map_180, '270': pred_map_270
    #     }
    #     patch_data['feat_shape'] = (H_feat, W_feat)

    #     # --- 2. Ensemble features ---
    #     ensembled_dino = torch.stack([dino_0, dino_90, dino_180, dino_270], dim=0).mean(dim=0)
    #     ensembled_clip = torch.stack([clip_0, clip_90, clip_180, clip_270], dim=0).mean(dim=0)
    #     patch_data['ensembled_dino_feat'] = ensembled_dino
    #     patch_data['ensembled_clip_feat'] = ensembled_clip
        
    #     return patch_data

    # def forward_slide(self, img, img_metas, stride=112, crop_size=224):
    #     """
    #     [최적화됨] 모든 GPU 연산을 한 번의 루프로 통합하여 런타임 최소화
    #     """
    #     printing = False

    #     if type(img) == list:
    #         img = img[0].unsqueeze(0)
        
    #     h_stride, w_stride = (stride, stride)
    #     h_crop, w_crop = (crop_size, crop_size)
    #     batch_size, _, h_img, w_img = img.shape
    #     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    #     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    #     device = img.device

    #     # ========================================================================
    #     # Phase 1: 모든 패치에 대한 정보 일괄 계산 (Pre-computation)
    #     # ========================================================================
    #     if printing: print("Phase 1: Pre-computing all data for every patch...")
        
    #     patch_data_cache = []
    #     patch_counter = 0

    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             y1, x1 = h_idx * h_stride, w_idx * w_stride
    #             y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
    #             y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]
                
    #             # pad image when (image_size % patch_size != 0)
    #             H, W = crop_img.shape[2:]
    #             pad = self.compute_padsize(H, W, self.patch_size[0])

    #             if any(pad):
    #                 padded_img = F.pad(crop_img, pad, mode='constant', value=0)
    #             else:
    #                 padded_img = crop_img
                
    #             # Compute in one-shot
    #             data = self._compute_patch_data(padded_img, (x1, y1, x2, y2), patch_counter, (H, W), pad)
    #             patch_data_cache.append(data)
    #             patch_counter += 1

    #     # ========================================================================
    #     # Phase 2: 강건한 앙상블 피처 풀 생성 (CPU-bound)
    #     # ========================================================================
    #     if printing: print("Phase 2: Building robust feature pool from cached data...")
        
    #     all_robust_dino, all_robust_clip, all_robust_patch_ids = [], [], []
        
    #     for data in patch_data_cache:
    #         maps = data['robustness_check_maps']
    #         robustness_mask = (maps['0'] == maps['90']) & (maps['0'] == maps['180']) & \
    #                           (maps['0'] == maps['270']) & (maps['0'] == maps['scale'])
            
    #         if robustness_mask.sum() > 0:
    #             dino_flat = data['ensembled_dino_feat'].flatten(2, 3).permute(0, 2, 1).reshape(-1, data['ensembled_dino_feat'].shape[1])
    #             clip_flat = data['ensembled_clip_feat'].flatten(2, 3).permute(2, 0, 1).reshape(-1, data['ensembled_clip_feat'].shape[0] * data['ensembled_clip_feat'].shape[1])
    #             mask_flat = robustness_mask.flatten()

    #             robust_dino_feats_patch = dino_flat[mask_flat]
    #             all_robust_dino.append(robust_dino_feats_patch)
    #             all_robust_clip.append(clip_flat[mask_flat])
    #             all_robust_patch_ids.append(torch.full((robust_dino_feats_patch.shape[0],), data['id'], device=device))

    #     if not all_robust_dino:
    #          all_robust_dino_feats = torch.empty(0, 768, device=device)
    #          all_robust_clip_feats = torch.empty(0, 768, device=device)
    #          all_robust_patch_ids = torch.empty(0, dtype=torch.long, device=device)
    #     else:
    #         all_robust_dino_feats = torch.cat(all_robust_dino, dim=0).to(dtype=torch.float32)
    #         all_robust_clip_feats = torch.cat(all_robust_clip, dim=0).to(dtype=torch.float32)
    #         all_robust_patch_ids = torch.cat(all_robust_patch_ids, dim=0)

    #     # ========================================================================
    #     # Phase 3: 동적 샘플링 및 최종 예측 (GPU 연산 최소화)
    #     # ========================================================================
    #     if printing: print("Phase 3: Final prediction with dynamic sampling and TTA...")
    #     preds = img.new_zeros((batch_size, self.num_queries, h_img, w_img))
    #     count_mat = img.new_zeros((batch_size, 1, h_img, w_img))

    #     for data in patch_data_cache:
    #         x1, y1, x2, y2 = data['coords']
            
    #         # --- 동적 외부 정보 구성 ---
    #         ref_dino, ref_clip = None, None
    #         external_mask = (all_robust_patch_ids != data['id'])
    #         external_dino_feats = all_robust_dino_feats[external_mask]
            
    #         if external_dino_feats.shape[0] > 0:
    #             external_clip_feats = all_robust_clip_feats[external_mask]

    #             K = 25; M = 80
    #             K = min(K, external_dino_feats.shape[0])
    #             kmeans = KMeans(n_clusters=K, init="k-means++", n_init=1, max_iter=25, random_state=42)
    #             labels = torch.tensor(kmeans.fit_predict(external_dino_feats.cpu().numpy()), device=device, dtype=torch.long)
    #             dino_samples, clip_samples = [], []
    #             for cid in range(K):
    #                 member_indices = (labels == cid).nonzero(as_tuple=True)[0]
    #                 if len(member_indices) == 0: continue
                    
    #                 if len(member_indices) > M:
    #                     rand_indices = member_indices[torch.randperm(len(member_indices), device=device)[:M]]
    #                 else:
    #                     rand_indices = member_indices[torch.randint(0, len(member_indices), (M,), device=device)]
                    
    #                 dino_samples.append(external_dino_feats[rand_indices])
    #                 clip_samples.append(external_clip_feats[rand_indices])
    #             if dino_samples:
    #                 ref_dino = torch.cat(dino_samples, dim=0).t().unsqueeze(0).contiguous()
    #                 ref_clip = torch.cat(clip_samples, dim=0).view(-1, self.n_head, self.head_dim).permute(1, 2, 0).contiguous()

    #         # --- 최종 예측 시 TTA 수행 ---
    #         crop_img = img[:, :, y1:y2, x1:x2] 
            
    #         H_crop_orig, W_crop_orig = crop_img.shape[-2:]
    #         pad = self.compute_padsize(H_crop_orig, W_crop_orig, self.patch_size[0])

    #         if any(pad):
    #             padded_crop_img = F.pad(crop_img, pad, mode='constant', value=0)
    #         else:
    #             padded_crop_img = crop_img

    #         #crop_seg_logit = self.forward_feature(crop_img, ref_dino, ref_clip)

    #         logit_0 = self.forward_feature(padded_crop_img, ref_dino, ref_clip, ex_feats=data['dino_feature']['0'])
            
    #         img_90 = TF.rotate(padded_crop_img, 90)
    #         logit_90_raw = self.forward_feature(img_90, ref_dino, ref_clip, ex_feats=data['dino_feature']['90'])
    #         logit_90 = torch.rot90(logit_90_raw, k=-1, dims=(2, 3)) # 결과 복원

    #         img_180 = TF.rotate(padded_crop_img, 180)
    #         logit_180_raw = self.forward_feature(img_180, ref_dino, ref_clip, ex_feats=data['dino_feature']['180'])
    #         logit_180 = torch.rot90(logit_180_raw, k=-2, dims=(2, 3)) # 결과 복원

    #         img_270 = TF.rotate(padded_crop_img, 270)
    #         logit_270_raw = self.forward_feature(img_270, ref_dino, ref_clip, ex_feats=data['dino_feature']['270'])
    #         logit_270 = torch.rot90(logit_270_raw, k=-3, dims=(2, 3)) # 결과 복원

    #         crop_seg_logit = torch.stack([logit_0, logit_90, logit_180, logit_270], dim=0).mean(dim=0)

    #         # --- 예측 결과에서 패딩 제거 ---
    #         if any(pad):
    #             l, _, t, _ = pad
    #             # forward_feature의 출력은 패딩된 입력 크기(padded_crop_img)와 동일
    #             # 따라서 원본 crop 크기(H_crop_orig, W_crop_orig)만큼 잘라냄
    #             crop_seg_logit = crop_seg_logit[:, :, t:t + H_crop_orig, l:l + W_crop_orig]

    #         preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
    #         count_mat[:, :, y1:y2, x1:x2] += 1

    #     assert (count_mat == 0).sum() == 0
    #     preds = preds / count_mat
    #     img_size = img_metas[0]['ori_shape'][:2]
    #     logits = F.interpolate(preds, size=img_size, mode='bilinear')

    #     return logits

    # Test
    # def forward_slide(self, img, img_metas, stride=112, crop_size=224):
    #     """
    #     [수정됨] 각 슬라이딩 패치에 대해 '자신을 제외한' 외부 정보로 클러스터링 및 샘플링 수행
    #     """
    #     printing = False

    #     if type(img) == list:
    #         img = img[0].unsqueeze(0)
        
    #     h_stride, w_stride = (stride, stride)
    #     h_crop, w_crop = (crop_size, crop_size)
    #     batch_size, _, h_img, w_img = img.shape
    #     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    #     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    #     device = img.device
        
    #     # ========================================================================
    #     # Phase 1: 강건한(Robust) 개별 피처 임베딩 및 '패치 ID' 수집
    #     # ========================================================================
    #     if printing: print("Phase 1: Collecting robust feature embeddings and their patch IDs...") 
        
    #     all_robust_dino = []
    #     all_robust_clip = []
    #     all_robust_patch_ids = [] 
    #     total_feature_patches = 0
    #     patch_counter = 0

    #     # pred_map_global_scaled = self._get_upscaled_global_pred_map(img)

    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             # ... (crop_img 생성 및 강건성 마스크 계산 로직은 동일) ...
    #             y1, x1 = h_idx * h_stride, w_idx * w_stride
    #             y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
    #             y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]

    #             pred_map_0 = self._predict_feature_map(crop_img)
    #             H_feat, W_feat = pred_map_0.shape
    #             total_feature_patches += H_feat * W_feat

    #             # pred_map_90 = torch.rot90(self._predict_feature_map(TF.rotate(crop_img, 90)), k=-1, dims=(0, 1))
    #             # pred_map_180 = torch.rot90(self._predict_feature_map(TF.rotate(crop_img, 180)), k=-2, dims=(0, 1))
    #             # pred_map_270 = torch.rot90(self._predict_feature_map(TF.rotate(crop_img, 270)), k=-3, dims=(0, 1))

    #             # y1_feat, x1_feat = y1 // self.dino_patch_size, x1 // self.dino_patch_size
    #             # y2_feat, x2_feat = y1_feat + H_feat, x1_feat + W_feat
    #             # pred_map_scale_crop = pred_map_global_scaled[y1_feat:y2_feat, x1_feat:x2_feat]

    #             # robustness_mask = (pred_map_0 == pred_map_90) & (pred_map_0 == pred_map_180) & \
    #             #                   (pred_map_0 == pred_map_270) #& (pred_map_0 == pred_map_scale_crop)
                
    #             # no augmentation
    #             robustness_mask = (pred_map_0 == pred_map_0)

    #             if robustness_mask.sum() > 0:
    #                 #dino_feat, clip_feat = self.ref_feature(crop_img)
    #                 dino_feat, clip_feat = self._get_ensembled_features(crop_img)

    #                 dino_feat_flat = dino_feat.flatten(2, 3).permute(0, 2, 1).reshape(-1, dino_feat.shape[1])
    #                 clip_feat_flat = clip_feat.flatten(2, 3).permute(2, 0, 1).reshape(-1, clip_feat.shape[0] * clip_feat.shape[1])
    #                 mask_flat = robustness_mask.flatten()
                    
    #                 robust_dino_feats_patch = dino_feat_flat[mask_flat]
    #                 all_robust_dino.append(robust_dino_feats_patch)
    #                 all_robust_clip.append(clip_feat_flat[mask_flat])

    #                 # [추가] 이 강건한 피처들이 현재 패치(patch_counter) 소속임을 기록
    #                 num_robust_in_patch = robust_dino_feats_patch.shape[0]
    #                 all_robust_patch_ids.append(torch.full((num_robust_in_patch,), patch_counter, device=device))

    #             patch_counter += 1
        
    #     # 수집된 모든 강건한 임베딩과 ID를 하나의 텐서로 통합
    #     if not all_robust_dino:
    #          all_robust_dino_feats = torch.empty(0, 768, device=device)
    #          all_robust_clip_feats = torch.empty(0, 768, device=device)
    #          all_robust_patch_ids = torch.empty(0, dtype=torch.long, device=device)
    #     else:
    #         all_robust_dino_feats = torch.cat(all_robust_dino, dim=0).to(dtype=torch.float32)
    #         all_robust_clip_feats = torch.cat(all_robust_clip, dim=0).to(dtype=torch.float32)
    #         all_robust_patch_ids = torch.cat(all_robust_patch_ids, dim=0)

    #     num_total_robust = all_robust_dino_feats.shape[0]
    #     if printing: print(f"  - Collected {num_total_robust} robust feature embeddings in total.")

    #     # ========================================================================
    #     # Phase 2 & 3: 동적 외부 정보 샘플링 및 최종 예측
    #     # ========================================================================
    #     if printing: print("Phase 2 & 3: Performing dynamic sampling and final prediction...")
    #     preds = img.new_zeros((batch_size, self.num_queries, h_img, w_img))
    #     count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    #     patch_counter = 0

    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             # --- 1. 현재 패치에 대한 '외부 정보 풀' 구성 ---
    #             ref_dino, ref_clip = None, None
                
    #             # 현재 패치 ID를 제외한 외부 임베딩만 선택
    #             external_mask = (all_robust_patch_ids != patch_counter)
    #             external_dino_feats = all_robust_dino_feats[external_mask]
    #             external_clip_feats = all_robust_clip_feats[external_mask]
                
    #             num_external_robust = external_dino_feats.shape[0]
                
    #             if num_external_robust > 0:
    #                 # --- 2. 외부 정보 풀에 대해 K-means 및 샘플링 수행 ---
    #                 K = 25
    #                 K = min(K, num_external_robust)
    #                 kmeans = KMeans(n_clusters=K, init="k-means++", n_init=1, max_iter=25, random_state=42)
    #                 labels = torch.tensor(kmeans.fit_predict(external_dino_feats.cpu().numpy()), device=device, dtype=torch.long)
                    
    #                 M = 80
    #                 dino_samples, clip_samples = [], []
    #                 for cid in range(K):
    #                     member_indices = (labels == cid).nonzero(as_tuple=True)[0]
    #                     if len(member_indices) == 0: continue
                        
    #                     if len(member_indices) > M:
    #                         rand_indices = member_indices[torch.randperm(len(member_indices), device=device)[:M]]
    #                     else:
    #                         rand_indices = member_indices[torch.randint(0, len(member_indices), (M,), device=device)]
                        
    #                     dino_samples.append(external_dino_feats[rand_indices])
    #                     clip_samples.append(external_clip_feats[rand_indices])
                    
    #                 if dino_samples:
    #                     ref_dino = torch.cat(dino_samples, dim=0).t().unsqueeze(0).contiguous()
    #                     ref_clip = torch.cat(clip_samples, dim=0).view(-1, 12, 64).permute(1, 2, 0).contiguous()

    #             # --- 3. 최종 예측 수행 ---
    #             y1, x1 = h_idx * h_stride, w_idx * w_stride
    #             y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
    #             y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]

    #             crop_seg_logit = self.forward_feature(crop_img, ref_dino, ref_clip)

    #             preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
    #             count_mat[:, :, y1:y2, x1:x2] += 1
                
    #             patch_counter += 1
            
    #     assert (count_mat == 0).sum() == 0
    #     preds = preds / count_mat
    #     img_size = img_metas[0]['ori_shape'][:2]
    #     logits = F.interpolate(preds, size=img_size, mode='bilinear')

    #     return logits


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