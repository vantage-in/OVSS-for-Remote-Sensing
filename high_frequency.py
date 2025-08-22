from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def extract_high_frequency_map_tensor(img_tensor: torch.Tensor, radius: int = 30) -> torch.Tensor:
    """
    Preprocessed image tensor에서 high-frequency map을 추출합니다.

    Args:
        img_tensor: torch.Tensor, shape [1, C, H, W], float, on cuda 혹은 cpu
        radius: int, low-frequency를 차단할 중앙 사각형 반지름 (픽셀 단위)

    Returns:
        high_freq_map: torch.Tensor, shape [1, C, H, W], 원본과 동일한 device에 반환
    """
    device = img_tensor.device
    x = img_tensor[0]  # [C, H, W]
    high_freq_channels = []

    for c in range(x.shape[0]):
        channel = x[c]  # [H, W]

        # 1) FFT → shift
        f      = torch.fft.fft2(channel)
        fshift = torch.fft.fftshift(f)

        # 2) 중앙 low-frequency 마스크 생성
        H, W = channel.shape
        crow, ccol = H // 2, W // 2
        mask = torch.ones((H, W), device=device, dtype=fshift.dtype)
        mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0

        # 3) mask 적용
        fshift_filt = fshift * mask

        # 4) 역 shift → 역 FFT → magnitude
        f_ishift = torch.fft.ifftshift(fshift_filt)
        img_back = torch.fft.ifft2(f_ishift)
        high_freq_channels.append(torch.abs(img_back))

    # 채널별 결과를 다시 스택
    high_freq_map = torch.stack(high_freq_channels, dim=0).unsqueeze(0)  # [1, C, H, W]
    return high_freq_map

def extract_bandpass_map_tensor(
    img_tensor: torch.Tensor,
    low_radius: int = 10,
    high_radius: int = 30
) -> torch.Tensor:
    """
    img_tensor에 band-pass filter를 적용하여 특정 주파수 대역만 남깁니다.

    Args:
        img_tensor: torch.Tensor, shape [1, C, H, W]
        low_radius: int, 차단할 저주파 대역의 최대 반지름 (픽셀 단위)
        high_radius: int, 차단할 고주파 대역의 최소 반지름 (픽셀 단위)

    Returns:
        bandpass_map: torch.Tensor, shape [1, C, H, W]
    """
    device = img_tensor.device
    x = img_tensor[0]            # [C, H, W]
    C, H, W = x.shape
    crow, ccol = H // 2, W // 2

    # 1) 중심으로부터의 거리 행렬 생성
    ys = torch.arange(H, device=device).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=device).view(1, W).expand(H, W)
    dist = torch.sqrt((ys - crow)**2 + (xs - ccol)**2)

    # 2) band-pass 마스크: low_radius ≤ dist ≤ high_radius
    mask = ((dist >= low_radius) & (dist <= high_radius)).to(torch.float32)

    bandpass_channels = []
    for c in range(C):
        channel = x[c]  # [H, W]

        # 3) FFT → shift
        fshift = torch.fft.fftshift(torch.fft.fft2(channel))

        # 4) mask 적용
        fshift_filt = fshift * mask

        # 5) 역 shift → 역 FFT → magnitude
        img_back = torch.fft.ifft2(torch.fft.ifftshift(fshift_filt))
        bandpass_channels.append(torch.abs(img_back))

    # 6) 다시 [1, C, H, W] 로 합치기
    bandpass_map = torch.stack(bandpass_channels, dim=0).unsqueeze(0)
    return bandpass_map

# ─── 사용 예시 ────────────────────────────────────────────────────────────────

img = Image.open('demo/dortmund_27.tif')
img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    transforms.Resize((448, 448))
])(img)
img_tensor = img_tensor.unsqueeze(0).to('cuda')

# 1) high-frequency 맵 추출
high_freq_tensor = extract_high_frequency_map_tensor(img_tensor, radius=30)
#high_freq_tensor = extract_bandpass_map_tensor(img_tensor, low_radius=10, high_radius=30)

# 2) 시각화를 위해 원래 이미지(unnormalize)와 고주파 맵을 CPU로 가져와 numpy 변환
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img_tensor.device).view(3,1,1)
std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img_tensor.device).view(3,1,1)

# 원본 unnormalize & clamp
orig = img_tensor[0] * std + mean
orig = torch.clamp(orig, 0.0, 1.0).permute(1,2,0).cpu().numpy()

# 고주파 맵 (채널별)
hf = high_freq_tensor[0].permute(1,2,0).cpu().numpy()
# 만약 그레이스케일로 보고 싶다면: hf_gray = hf.mean(axis=2)

# 3) matplotlib으로 출력
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Original (Unnormalized)')
plt.imshow(orig)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('High-Frequency Map')
plt.imshow(hf.mean(axis=2), cmap='gray')  # 채널 평균한 흑백 맵
plt.axis('off')

plt.savefig('visualize/hf.png', bbox_inches='tight')

# 2) 그레이스케일: 채널 평균
hf_gray = hf.mean(axis=2)  # (H, W)

# 3) 중간 행 인덱스
#mid = hf_gray.shape[0] // 2
mid = int(hf_gray.shape[0] * 0.9)

# 4) 중간 행 픽셀값 벡터 추출
mid_gray = hf_gray[mid]
mid_r    = hf[mid, :, 0]
mid_g    = hf[mid, :, 1]
mid_b    = hf[mid, :, 2]

# 5) 하나의 그래프에 색 지정하여 그리기
plt.figure(figsize=(12, 6))
plt.plot(mid_gray, label='Gray (Avg of R,G,B)', color='black', linewidth=2)
plt.plot(mid_r,    label='Channel R',             color='red',   linestyle='--')
plt.plot(mid_g,    label='Channel G',             color='green', linestyle='-.')
plt.plot(mid_b,    label='Channel B',             color='blue',  linestyle=':')

plt.title('Middle Row High-Frequency Map by Channel')
plt.xlabel('Column Index')
plt.ylabel('Magnitude')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('visualize/hf_graph.png', bbox_inches='tight')