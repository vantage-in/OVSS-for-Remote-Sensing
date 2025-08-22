"""
    python demo_hf.py demo/image/kyoto_33.tif
"""
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from segearth_segmentor import SegEarthSegmentation
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.patches as mpatches
import torch, torchvision
import torch.nn.functional as F
from typing import List, Tuple
from torchvision.transforms import GaussianBlur

SAVE_DIR = Path('visualize/variants'); SAVE_DIR.mkdir(exist_ok=True, parents=True)
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]
NORM = transforms.Normalize(MEAN, STD)

# ──────────────────────────────────────────────────────────────
# 1. FFT‑based high / low pass helpers
# ──────────────────────────────────────────────────────────────
def fft_filter(img_t: torch.Tensor, radius: int, keep_low: bool) -> torch.Tensor:
    """FFT helper: preserve low (LPF) or high (HPF) within given radius.
    img: [1,C,H,W] float 0‑1, radius in pixels"""
    B, C, H, W = img_t.shape
    device, dtype = img_t.device, img_t.dtype
    out = torch.zeros_like(img_t)

    # 마스크 (동일 H,W 사용)
    Y, X = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device), indexing='ij')
    dist2 = (Y - H//2)**2 + (X - W//2)**2
    mask = dist2 <= radius**2   # True == low-freq

    for c in range(C):
        f = torch.fft.fftshift(torch.fft.fft2(img_t[0,c]))
        if keep_low:
            f = f * mask        # LPF
        else:
            f = f * (~mask)     # HPF
        img_back = torch.fft.ifft2(torch.fft.ifftshift(f)).real
        out[0, c] = img_back.clip(0, 1)  # clip to 0-1 for saving
    return out

def high_pass(img_t, r=10): return fft_filter(img_t, r, keep_low=False)
def low_pass (img_t, r=100): return fft_filter(img_t, r, keep_low=True)

# ──────────────────────────────────────────────────────────────
# 2. Spatial‑domain Gaussian blur & Unsharp mask
# ──────────────────────────────────────────────────────────────
def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """RGB Gaussian blur with given σ."""
    g = GaussianBlur(kernel_size=int(max(3, 4 * sigma + 1)), sigma=sigma)
    return g(img)
def unsharp_mask(img: torch.Tensor, sigma: float = 1.0, amount: float = 1.5) -> torch.Tensor:
    """Unsharp mask: sharpen image by adding scaled high‑freq component."""
    blurred = gaussian_blur(img, sigma)
    return (img + amount * (img - blurred)).clamp(0, 1)


def gaussian_noise(img: torch.Tensor, std: float) -> torch.Tensor:
    orig_dtype = img.dtype
    img_f = img.float()                       # float32 변환
    noise = torch.randn_like(img_f) * std     # 가우시안 노이즈
    noised = img_f + noise

    # 값 범위 고정 (uint8 이미지면 0-255, 정규화(0-1) 이미지면 0-1 등 상황에 맞춰 조정)
    lo, hi = (0.0, 255.0) if orig_dtype == torch.uint8 else (0.0, 1.0)
    noised = noised.clamp(lo, hi)

    return noised.to(orig_dtype)              # 원래 dtype으로 캐스팅

# ──────────────────────────────────────────────
# 2-bis. Grid-mask helper  (14×14 패치 기준)
# ──────────────────────────────────────────────
def grid_zero_mask(img: torch.Tensor,
                   grid: int = 14,
                   prob: float = 0.3) -> torch.Tensor:
    """
    img  : [1,C,H,W] tensor, 값 범위 0-1 (정규화 이전)
    grid : 한 변의 패치 수 (14이면 14×14 = 196패치)
    prob : 각 패치가 0으로 가려질 확률 (0~1)
    returns: 마스킹된 이미지 (same shape)
    """
    _, C, H, W = img.shape
    ph, pw = H // grid, W // grid             # 패치 크기
    img = img.clone()

    for gy in range(grid):
        for gx in range(grid):
            if torch.rand((42)) < prob:
                top, left = gy * ph, gx * pw
                img[:, :, top:top+ph, left:left+pw] = 0.0
    return img

# ──────────────────────────────────────────────────────────────
# 3. Variant preparation util
# ──────────────────────────────────────────────────────────────
def _save_img(x: torch.Tensor, stem: str):
    """Stretch [0,1] tensor and save as PNG."""
    x = x.squeeze(0)
    stretched = (x - x.min()) / (x.max() - x.min() + 1e-8)
    torchvision.utils.save_image(stretched, str(SAVE_DIR / f"{stem}.png"))

def prepare_variant(img_path: str, variant: str, **params) -> Tuple[torch.Tensor, Image.Image, str]:
    """Create variant, save PNG, and return normalised tensor ready for model.

    variant: "high" | "low" | "blur" | "sharp"
    params: radius_h / radius_l / sigma / amount according to variant
    """
    # 1) load & resize
    base = Image.open(img_path).convert("RGB").resize((448, 448), Image.Resampling.BILINEAR)
    x = transforms.ToTensor()(base).unsqueeze(0)  # [1,C,H,W] 0-1

    # 2) apply filter
    if variant == "high":
        radius = int(params.get("radius_h", 30))
        x_var = high_pass(x, radius)
        suffix = f"high_r{radius}"
    elif variant == "low":
        radius = int(params.get("radius_l", 30))
        x_var = low_pass(x, radius)
        suffix = f"low_r{radius}"
    elif variant == "blur":
        sigma = float(params.get("sigma", 1.0))
        x_var = gaussian_blur(x, sigma)
        suffix = f"blur_s{str(sigma).replace('.', '-')}"
    elif variant == "sharp":
        sigma = float(params.get("sigma", 1.0))
        amount = float(params.get("amount", 1.5))
        x_var = unsharp_mask(x, sigma, amount)
        suffix = f"sharp_s{str(sigma).replace('.', '-')}_a{str(amount).replace('.', '-')}"
    elif variant == "noise":
        std = float(params.get("std", 10.0))
        x_var = gaussian_noise(x, std)
        suffix = f"noise_s{str(std).replace('.', '-')}"
    elif variant == "gridmask":
        p = float(params.get("prob", 0.3))        # 마스킹 확률
        g = int(params.get("grid", 14))           # 패치 그리드 크기
        x_var = grid_zero_mask(x, grid=g, prob=p)
        suffix = f"gridmask_g{g}_p{str(p).replace('.','-')}"
    else:
        raise ValueError("variant must be high | low | blur | sharp | noise | gridmask")

    # 3) save PNG for visual check
    stem = f"{Path(img_path).stem}_{suffix}"
    _save_img(x_var, stem)

    # 4) normalise for model input
    x_norm = NORM(x_var.squeeze(0)).unsqueeze(0) # [1,C,H,W], ready for SegEarthSegmentation

    return x_norm, base, suffix

# ──────────────────────────────────────────────────────────────
# 4. Batch runner for quick experiments
# ──────────────────────────────────────────────────────────────
RADII_HP: List[int] = [1, 2, 5, 10, 20, 30, 50, 100]
RADII_LP: List[int] = [5, 10, 20, 30, 50, 75, 100, 150]
SIGMAS_BLUR: List[float] = [2, 4, 8]
UNSHARP_PARAM: List[Tuple[float, float]] = [(1, 0.5), (1, 1.0), (1, 2.0), (2, 1.5), (2, 3.0), (2, 10.0), (4, 2.0), (4, 4.0), (4, 10.0)]
STD_NOISE: List[float] = [0.05, 0.2, 0.5]
GRID_P_LIST = [0.1, 0.3, 0.5]   # 원하는 확률 목록

# ──────────────────────────────────────────────────────────────
# 5. Functions to save visualizations
# ──────────────────────────────────────────────────────────────
class_info = {
    0: ("Background",        "#000000"),  # black
    1: ("Bareland",          "#800000"),
    2: ("Rangeland (grass)", "#00FF24"),
    3: ("Developed space (pavement)",   "#949494"),
    4: ("Road",              "#FFFFFF"),
    5: ("Tree",              "#226126"),
    6: ("Water",             "#0045FF"),
    7: ("Agriculture land (cropland)",  "#4BB549"),
    8: ("Building",          "#DE1F07"),
}
colors = [mcolors.hex2color(class_info[i][1]) for i in range(9)]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(10) - 0.5, cmap.N)

def save_mask_with_legend(mask, filename, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mask, cmap=cmap, norm=norm)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(9),
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([class_info[i][0] for i in range(9)])
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)

def save_overlay_with_legend(img, seg_mask, filename, name_list):
    # 1) 준비
    fig, ax = plt.subplots(figsize=(8, 8))
    # 2) 원본 리사이즈
    resized_img = img.resize((448, 448), resample=Image.Resampling.BILINEAR)
    ax.imshow(resized_img)
    # 3) 분할 colormap
    cmap = plt.get_cmap('tab20', len(name_list))
    ax.imshow(seg_mask,
              cmap=cmap,
              alpha=0.5,
              vmin=0, vmax=len(name_list)-1)

    # 5) legend 추가
    patches = []
    for idx, full_name in enumerate(name_list):
        label = full_name.split(',')[0]
        color = cmap(idx)
        patches.append(mpatches.Patch(color=color, label=label))
    ax.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1.0),
        loc='upper left',
        fontsize=8,
        frameon=False
    )

    # 6) 마무리 저장
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

def run_one_variant(tensor: torch.Tensor,
                    base_img: Image.Image,
                    suffix: str,
                    img_stem: str,
                    save_dir: Path,
                    *,
                    model=None,
                    device="cpu",
                    name_list=None):
    """Save variant PNGs + segmentation (if model given)."""
    # save filter PNG for visual inspection
    torchvision.utils.save_image(tensor.clone().cpu().squeeze(0) * torch.tensor(STD).view(3,1,1) + torch.tensor(MEAN).view(3,1,1),
                                 str(save_dir / f"{img_stem}_{suffix}.png"))

    if model is None:
        return

    with torch.no_grad():
        seg = model.predict(tensor.to(device), data_samples=None)

    if seg.ndim == 4:  # [B,C,H,W] logits
        mask = torch.argmax(seg, dim=1).squeeze(0).cpu().numpy()
    else:              # [B,H,W] already mask
        mask = seg.squeeze(0).cpu().numpy()

    save_mask_with_legend(
        mask, 
        save_dir / f"{img_stem}_{suffix}_mask.png", 
        f"OpenEarthMap - {img_stem}_{str(suffix).replace('-', '.')}"
        )
    save_overlay_with_legend(
        base_img, 
        mask, 
        save_dir / f"{img_stem}_{suffix}_overlay.png", 
        name_list
        )

# ──────────────────────────────────────────────────────────────
# 6. Example usage (SegEarth model inference commented)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate filter variants for SegEarth experiments")
    parser.add_argument("image", help="Path to source RGB image")
    parser.add_argument("--device", default="cuda", help="cpu | cuda")
    args = parser.parse_args()

    img_stem = Path(args.image).stem

    # 2) Load model -----------------------------------------------------------
    model = SegEarthSegmentation(
        clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
        vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
        model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
        ignore_residual=True,
        feature_up=False,
        feature_up_cfg=dict(
            model_name='jbu_one',
            model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
        cls_token_lambda=-0.0,
        name_path='./configs/my_name.txt',
        prob_thd=0.1,
    ).to(args.device).eval()

    name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
             'tree,forest', 'water,river', 'cropland', 'building,roof,house']

    with open('./configs/my_name.txt', 'w') as writers:
        for i in range(len(name_list)):
            if i == len(name_list)-1:
                writers.write(name_list[i])
            else:
                writers.write(name_list[i] + '\n')
    writers.close()


    
    # (1) Generate tensors -----------------------------------------------------
    # for r in RADII_HP:
    #     tensor, base, suf = prepare_variant(args.image, "high", radius_h=r)
    #     run_one_variant(tensor, base, suf, img_stem, SAVE_DIR, model=model, device=args.device, name_list=name_list)

    # for r in RADII_LP:
    #     tensor, base, suf = prepare_variant(args.image, "low", radius_l=r)
    #     run_one_variant(tensor, base, suf, img_stem, SAVE_DIR, model=model, device=args.device, name_list=name_list)

    for s in SIGMAS_BLUR:
        tensor, base, suf = prepare_variant(args.image, "blur", sigma=s)
        #run_one_variant(tensor, base, suf, img_stem, SAVE_DIR, model=model, device=args.device, name_list=name_list)
        row, col = 420, 365
        logits = model.predict_logit(tensor.to(args.device), data_samples=None)[0]
        pixel_logits = logits[:, row, col]     # 확률 합 = 1
        print(f"\n[Pixel ({row},{col}) class logits: Blur {suf}]")
        for idx, p in enumerate(pixel_logits):
            print(f"{idx:2d} – {name_list[idx]:25s}: {p.item()*100:6.2f}")

    # for s, a in UNSHARP_PARAM:
    #     tensor, base, suf = prepare_variant(args.image, "sharp", sigma=s, amount=a)
    #     run_one_variant(tensor, base, suf, img_stem, SAVE_DIR, model=model, device=args.device, name_list=name_list)
    #     # row, col = 210, 225
    #     # logits = model.predict_logit(tensor.to(args.device), data_samples=None)[0]
    #     # pixel_logits = logits[:, row, col]     # 확률 합 = 1
    #     # print(f"\n[Pixel ({row},{col}) class logits: Sharp {suf}]")
    #     # for idx, p in enumerate(pixel_logits):
    #     #     print(f"{idx:2d} – {name_list[idx]:25s}: {p.item()*100:6.2f}")    
    
    for s in STD_NOISE:
        tensor, base, suf = prepare_variant(args.image, "noise", std=s)
        #run_one_variant(tensor, base, suf, img_stem, SAVE_DIR, model=model, device=args.device, name_list=name_list)
        row, col = 420, 365
        logits = model.predict_logit(tensor.to(args.device), data_samples=None)[0]
        pixel_logits = logits[:, row, col]     # 확률 합 = 1
        print(f"\n[Pixel ({row},{col}) class logits: Noise {suf}]")
        for idx, p in enumerate(pixel_logits):
            print(f"{idx:2d} – {name_list[idx]:25s}: {p.item()*100:6.2f}")

    # for p in GRID_P_LIST:
    #     tensor, base, suf = prepare_variant(
    #         args.image, "gridmask", prob=p, grid=14)
    #     run_one_variant(tensor, base, suf, img_stem, SAVE_DIR,
    #                     model=model, device=args.device, name_list=name_list)

    # Subtract 해보기
    # tensor = ...
    # logits = model.predict_logit(tensor, data_samples=None)
    # logits_noup = model_noup.predict_logit(tensor, data_samples=None)

    # tensor_var, base, suf = prepare_variant(args.image, "blur", sigma=4.0)
    # logits_var = model.predict_logit(tensor_var.to(args.device), data_samples=None)
    # logits_var_noup = model_noup.predict_logit(tensor_var.to(args.device), data_samples=None)

    # seg_logits = logits - (logits_var - logits_var_noup)
    # seg_pred = model.postprocess_result(seg_logits, data_samples=None)

    # save_mask_with_legend(mask, name_list, SAVE_DIR / f"{img_stem}_{suffix}_mask.png")
    # save_overlay_with_legend(base_img, mask, name_list, SAVE_DIR / f"{img_stem}_{suffix}_overlay.png")



    #     # Prepare high‑pass variant and run prediction
    # variant = 'low'
    # img_tensor = prepare_variant(img_path, variant, radius=r).to('cuda')
    # with torch.no_grad():
    #     seg_pred = model.predict(img_tensor, data_samples=None)
    # seg_mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

    # # Save outputs with radius suffix
    # pred_file = SAVE_DIR / f"{base_name}_pred_{variant}_r{r}.png"
    # overlay_file = SAVE_DIR / f"{base_name}_pred_overlay_{variant}_r{r}.png"

    # save_mask_with_legend(seg_mask, filename=str(pred_file),
    #                         title=f"OpenEarthMap - {base_name} ({variant}-PF r={r})")
    # save_overlay_with_legend(img, seg_mask, filename=str(overlay_file),
    #                             name_list=name_list)

    # print(f"[✓] Saved results for radius {r} → {pred_file}")


    # Example: feed into SegEarth model (uncomment when model available)
    # model = SegEarthSegmentation(...).to(args.device).eval()
    # with torch.no_grad():
    #     for name, t in tensors.items():
    #         pred = model.predict(t.to(args.device), data_samples=None)
    #         torch.save(pred, SAVE_DIR / f"{Path(args.image).stem}_{name}_seg.pt")




# img_path = 'demo/image/kyoto_33.tif'
# img = Image.open(img_path)
# base_name = Path(img_path).stem  # 'kyoto_33'

# name_list = ['background', 'bareland,barren', 'grass', 'pavement', 'road',
#              'tree,forest', 'water,river', 'cropland', 'building,roof,house']

# with open('./configs/my_name.txt', 'w') as writers:
#     for i in range(len(name_list)):
#         if i == len(name_list)-1:
#             writers.write(name_list[i])
#         else:
#             writers.write(name_list[i] + '\n')
# writers.close()

# img_tensor = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
#     transforms.Resize((448, 448))
# ])(img)


# model = SegEarthSegmentation(
#     clip_type='CLIP',     # 'CLIP', 'BLIP', 'OpenCLIP', 'MetaCLIP', 'ALIP', 'SkyCLIP', 'GeoRSCLIP', 'RemoteCLIP'
#     vit_type='ViT-B/16',      # 'ViT-B/16', 'ViT-L-14'
#     model_type='SegEarth',   # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
#     ignore_residual=True,
#     feature_up=True,
#     feature_up_cfg=dict(
#         model_name='jbu_one',
#         model_path='simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt'),
#     cls_token_lambda=-0.3,
#     name_path='./configs/my_name.txt',
#     prob_thd=0.1,
# )


# # === 6. Define the OpenEarthMap 9-class color map ===
# class_info = {
#     0: ("Background",        "#000000"),  # black
#     1: ("Bareland",          "#800000"),
#     2: ("Rangeland (grass)", "#00FF24"),
#     3: ("Developed space (pavement)",   "#949494"),
#     4: ("Road",              "#FFFFFF"),
#     5: ("Tree",              "#226126"),
#     6: ("Water",             "#0045FF"),
#     7: ("Agriculture land (cropland)",  "#4BB549"),
#     8: ("Building",          "#DE1F07"),
# }
# colors = [mcolors.hex2color(class_info[i][1]) for i in range(9)]
# cmap_9 = mcolors.ListedColormap(colors)
# norm_9 = mcolors.BoundaryNorm(np.arange(10) - 0.5, cmap_9.N)

# # === 7. Functions to save visualizations ===
# def save_mask_with_legend(mask, filename, title):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     im = ax.imshow(mask, cmap=cmap_9, norm=norm_9)
#     ax.axis("off")
#     ax.set_title(title, fontsize=14)
#     cbar = plt.colorbar(im, ax=ax, ticks=np.arange(9),
#                         fraction=0.046, pad=0.04)
#     cbar.ax.set_yticklabels([class_info[i][0] for i in range(9)])
#     cbar.ax.tick_params(labelsize=8)
#     plt.tight_layout()
#     plt.savefig(filename, bbox_inches="tight", dpi=300)
#     plt.close(fig)

# def save_overlay_with_legend(img, seg_mask, filename, name_list):
#     # 1) 준비
#     fig, ax = plt.subplots(figsize=(8, 8))
#     # 2) 원본 리사이즈
#     resized_img = img.resize((448, 448), resample=Image.Resampling.BILINEAR)
#     ax.imshow(resized_img)
#     # 3) 분할 colormap
#     cmap = plt.get_cmap('tab20', len(name_list))
#     ax.imshow(seg_mask,
#               cmap=cmap,
#               alpha=0.5,
#               vmin=0, vmax=len(name_list)-1)

#     # 5) legend 추가
#     patches = []
#     for idx, full_name in enumerate(name_list):
#         label = full_name.split(',')[0]
#         color = cmap(idx)
#         patches.append(mpatches.Patch(color=color, label=label))
#     ax.legend(
#         handles=patches,
#         bbox_to_anchor=(1.05, 1.0),
#         loc='upper left',
#         fontsize=8,
#         frameon=False
#     )

#     # 6) 마무리 저장
#     ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(filename, bbox_inches='tight', dpi=300)
#     plt.close(fig)

# # def save_overlay(original_img, mask, filename):
# #     fig, ax = plt.subplots(figsize=(8, 8))
# #     resized = original_img.resize(mask.shape[::-1],
# #                                   resample=Image.Resampling.BILINEAR)
# #     ax.imshow(resized)
# #     ax.imshow(mask, cmap=cmap_9, norm=norm_9, alpha=0.25)
# #     ax.axis("off")
# #     plt.tight_layout()
# #     plt.savefig(filename, bbox_inches="tight", dpi=300)
# #     plt.close(fig)

# RADII = [1, 2, 5, 10, 20, 30, 50, 100]
# RADII = [1, 5, 10, 20, 30, 50, 75, 100]
# for r in RADII:
#     # Prepare high‑pass variant and run prediction
#     variant = 'low'
#     img_tensor = prepare_variant(img_path, variant, radius=r).to('cuda')
#     with torch.no_grad():
#         seg_pred = model.predict(img_tensor, data_samples=None)
#     seg_mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

#     # Save outputs with radius suffix
#     pred_file = SAVE_DIR / f"{base_name}_pred_{variant}_r{r}.png"
#     overlay_file = SAVE_DIR / f"{base_name}_pred_overlay_{variant}_r{r}.png"

#     save_mask_with_legend(seg_mask, filename=str(pred_file),
#                             title=f"OpenEarthMap - {base_name} ({variant}-PF r={r})")
#     save_overlay_with_legend(img, seg_mask, filename=str(overlay_file),
#                                 name_list=name_list)

#     print(f"[✓] Saved results for radius {r} → {pred_file}")

# # #img_tensor = img_tensor.unsqueeze(0).to('cuda')
# # variant = 'high'
# # # x_high  = prepare_variant(img_path, 'high',  radius=30)
# # # x_low   = prepare_variant(img_path, 'low',   radius=30)
# # # x_sharp = prepare_variant(img_path, 'sharp', sigma=1.0, amount=1.5)

# # img_tensor = prepare_variant(img_path, variant).to('cuda')

# # seg_pred = model.predict(img_tensor, data_samples=None)
# # seg_mask = seg_pred.data.cpu().numpy().squeeze(0).astype(np.uint8)

# # # === 8. Save the outputs ===
# # if model.feature_up:
# #     pred_path = Path("visualize/hf") / f"{base_name}_pred_{variant}.png"
# #     overlay_path = Path("visualize/hf") / f"{base_name}_pred_overlay_{variant}.png"
# # else:
# #     pred_path = Path("visualize/hf") / f"{base_name}_no_pred.png"
# #     overlay_path = Path("visualize/hf") / f"{base_name}_no_pred_overlay.png"

# # save_mask_with_legend(
# #     seg_mask,
# #     filename=str(pred_path),
# #     title=f"OpenEarthMap - {base_name}"
# # )
# # save_overlay_with_legend(
# #     img,
# #     seg_mask,
# #     filename=f"visualize/hf/{base_name}_pred_overlay_{variant}.png",
# #     name_list=name_list
# # )
# # # save_overlay(
# # #     img,
# # #     seg_mask,
# # #     filename=str(overlay_path)
# # # )