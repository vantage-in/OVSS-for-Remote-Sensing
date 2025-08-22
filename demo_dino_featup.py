import os, torch, numpy as np, torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torch import Tensor
from segearth_segmentor import SegEarthSegmentation
from simfeatup_dev.upsamplers import get_upsampler

# ─── 설정 ───────────────────────────────────────────────────────
device      = "cuda"
model_name  = "dinov2"         # 'dino16' or 'dinov2'
coord_list = [(40, 80), (40, 180), (80, 60), (80, 150), (120, 70), (120, 190), (180, 100), (180, 180), (200, 20), (200, 200)]
img_path    = "demo/image/kyoto_33.tif"
crop_pos    = "tl"             # 'tl','tr','bl','br'  (top-left 기본)

# 모델별 해상도 설정
inp_H = 224                    # 모델 입력은 항상 224
hr_H  = 224 if model_name=="dino16" else 256
patch_sz_lr = 16 if model_name=="dino16" else 14

# ─── 1. 이미지 로드 & 4-분할 crop → 224×224 ─────────────────────
img_rgb = Image.open(img_path).convert("RGB")
W, H    = img_rgb.size
half_W, half_H = W//2, H//2

# crop 좌표 결정
pos_map = {
    "tl": (0,         0),
    "tr": (half_W,    0),
    "bl": (0,       half_H),
    "br": (half_W, half_H)
}
if crop_pos not in pos_map:
    raise ValueError("crop_pos must be 'tl','tr','bl' or 'br'")
left, top = pos_map[crop_pos]
crop_box  = (left, top, left+half_W, top+half_H)   # (l,u,r,b)

crop_img = img_rgb.crop(crop_box)                  # PIL.Image
tfm = transforms.Compose([
    transforms.Resize((inp_H, inp_H), antialias=True),
    transforms.ToTensor()
])
img_tensor = tfm(crop_img).unsqueeze(0).to(device)   # [1,3,224,224]
img_disp224 = img_tensor[0].cpu().permute(1,2,0).numpy()

# ─── 2. FeatUp 업샘플러 로드 ────────────────────────────────────
upsampler = torch.hub.load("mhamilton723/FeatUp",
                           model_name, use_norm=True
                          ).to(device).eval()

with torch.no_grad():
    hr_feats = upsampler(img_tensor)            # (1,C,hr_H,hr_H)
    lr_feats = upsampler.model(img_tensor)      # (1,C,h,w)

w_lr      = lr_feats.size(-1)                   # 14 or 16
patch_sz  = inp_H // w_lr                       # 16 or 14

# ----------- SegEarth-OV + SimFeatUp --------------------
seg_img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((448, 448))
])(img_rgb).unsqueeze(0).to('cuda')
top, left = 0, 0
height, width = 224, 224
seg_img_tensor = seg_img_tensor[:, :, top:top+height, left:left+width].half()  # [B, C, 224, 224]

model = SegEarthSegmentation(
    clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
    vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
    model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
    ignore_residual=True,
    feature_up=True,
    feature_up_cfg=dict(
        model_name='jbu_one',
        model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
    cls_token_lambda=-0.0,
    name_path='./configs/my_name.txt',
    prob_thd=0.1,
    cls_variant="none",
)
image_features = model.net.encode_image(seg_img_tensor, model.model_type, model.ignore_residual, model.output_cls_token)

simfeatup = get_upsampler('jbu_one', 512).cuda().half()
ckpt = torch.load('simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt')['state_dict']
weights_dict = {k[10:]: v for k, v in ckpt.items()}
simfeatup.load_state_dict(weights_dict, strict=True)

feature_w, feature_h = seg_img_tensor[0].shape[-2] // 16, seg_img_tensor[0].shape[-1] // 16
image_w, image_h = seg_img_tensor[0].shape[-2], seg_img_tensor[0].shape[-1]
image_features = image_features.permute(0, 2, 1).view(1, 512, feature_w, feature_h)
with torch.no_grad():
    with torch.cuda.amp.autocast():
        sim_hr_feats = simfeatup(image_features, seg_img_tensor).half()
sim_hr_feats = sim_hr_feats.view(1, 512, image_w, image_h)

# ─── 3. similarity → heat map 함수 ─────────────────────────────
def sim_map(feats: Tensor, q:int)->Tensor:
    B,C,H,W = feats.shape
    x = F.normalize(feats.reshape(B,C,H*W), dim=1)
    attn = torch.einsum("b c m, b c n->b m n", x, x)
    attn[attn<0.3] = float('-inf')
    attn = F.softmax(attn, -1)[0,q].reshape(H,W)
    return (attn/attn.max().clamp(1e-8)).detach().cpu()

def lr_idx_to_hr(idx:int)->int:
    r,c = divmod(idx,w_lr)
    return (r*16+16//2)*hr_H + (c*16+16//2)

def coord_to_qhr(y_pix: int, x_pix: int) -> tuple[int, int, int]:
    """
    y_pix, x_pix : 224×224 기준 픽셀 좌표
    returns      : (q_lr, q_hr, (y_hr, x_hr))
      q_lr : LR grid 1-D index
      q_hr : HR grid 1-D index (224 or 256²)
      (y_hr,x_hr) : HR 2-D 위치 (마커용)
    """
    # 1) 224 좌표 → LR 패치 인덱스
    r_lr, c_lr = y_pix // patch_sz, x_pix // patch_sz
    q_lr = r_lr * w_lr + c_lr

    # 2) LR 인덱스 → HR 패치 중심
    q_hr_center = lr_idx_to_hr(q_lr)            # 1-D (HR)
    y_center, x_center = divmod(q_hr_center, hr_H)

    # 3) HR=224 ⟹ q_224 = q_hr_center (동일 인덱스)
    if hr_H == inp_H:                           # dino16
        q_224 = q_hr_center
        return q_224, q_hr_center, (y_center, x_center)

    # 4) HR=256 ⟹ HR 중심을 224 스케일로 역-사상
    y_224 = int(round(y_center * inp_H / hr_H))   # inp_H=224
    x_224 = int(round(x_center * inp_H / hr_H))
    q_224 = y_224 * inp_H + x_224                # 1-D in 224 grid
    return q_224, q_hr_center, (y_center, x_center)

lr_feats_up = F.interpolate(lr_feats, size=(inp_H,inp_H),
                            mode='bilinear', align_corners=False)

# ─── 4. 반복 시각화 & 저장 ────────────────────────────────────
os.makedirs("visualize/featup", exist_ok=True)
base = Path(img_path).stem

for (y_pix,x_pix) in coord_list:
    q_224, q_hr, (y_hr, x_hr) = coord_to_qhr(y_pix, x_pix)

    heat_lr = sim_map(lr_feats_up, q_224)    # LR→HR 이후 유사도
    heat_hr = sim_map(hr_feats,    q_hr)
    heat_sim = sim_map(sim_hr_feats, q_224)

    # overlay용 이미지 224→256 resize (dinov2만 해당)
    if hr_H != inp_H:
        img_disp = F.interpolate(img_tensor, size=(hr_H,hr_H),
                                 mode='bilinear', align_corners=False
                                )[0].cpu().permute(1,2,0).numpy()
    else:
        img_disp = img_disp224

    fig, ax = plt.subplots(1,4,figsize=(16,4),dpi=120)
    ax[0].imshow(img_disp);                 ax[0].set_title("Image")
    ax[0].plot(x_hr, y_hr, 'rx', ms=8, mew=2) 
    ax[1].imshow(img_disp); ax[1].imshow(heat_lr,alpha=.7,cmap='jet')
    ax[1].set_title(f"{model_name} + Bilinear : ({y_pix}, {x_pix})")
    ax[2].imshow(img_disp); ax[2].imshow(heat_hr,alpha=.7,cmap='jet')
    ax[2].set_title(f"{model_name} + FeatUp : ({y_pix}, {x_pix})")
    ax[3].imshow(img_disp); ax[3].imshow(heat_sim,alpha=.7,cmap='jet')
    ax[3].set_title(f"SegEarth + SimFeatUp : ({y_pix}, {x_pix})")
    for a in ax: a.axis('off')
    plt.tight_layout()

    out = f"visualize/featup/{base}_{crop_pos}_{model_name}_{y_pix}_{x_pix}.png"
    plt.savefig(out, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print("saved →", out)