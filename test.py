import cv2
import matplotlib.pyplot as plt
import numpy as np
# 1. 이미지 불러오기 (컬러 → 흑백 변환)
# img = cv2.imread("demo/image/kyoto_33.tif")#[:224,224:,:]  # 파일 경로 수정
# img = cv2.resize(img, (448,448), interpolation=cv2.INTER_LINEAR)[:224,:224,:]
img = cv2.imread("temp/unnorm_patch.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Gaussian Blur 두 번 적용 (σ1 < σ2)
blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)

# 3. DoG 계산
dog = blur1 - blur2

e_hf  = np.sum(dog**2)
e_tot = np.sum(gray**2) + 1e-8
hf_score = float(e_hf / e_tot) 

print(e_hf)
print(e_tot)

print(f"High-Frequency Energy Ratio: {hf_score:.4f}")

# # 4. 시각화
# plt.figure(figsize=(12,4))
# plt.subplot(1,3,1); plt.title("Original Gray"); plt.imshow(gray, cmap="gray"); plt.axis("off")
# plt.subplot(1,3,2); plt.title("DoG (σ=1-2)"); plt.imshow(dog, cmap="gray"); plt.axis("off")
# plt.subplot(1,3,3); plt.title("Absolute DoG"); plt.imshow(cv2.convertScaleAbs(dog), cmap="gray"); plt.axis("off")
# plt.show()