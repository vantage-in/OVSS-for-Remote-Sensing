import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from pathlib import Path

# === 사용자 입력 ===
label_path = "./demo/label/dolnoslaskie_27.tif"  # 단일 채널 label mask 경로

# === 클래스 정의 및 색상 정보 (OpenEarthMap 기준) ===
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

# === 시각화를 위한 color map 생성 ===
colors = [mcolors.hex2color(class_info[i][1]) for i in range(9)]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(np.arange(10) - 0.5, cmap.N)

# === label mask 불러오기 ===
label_img = Image.open(label_path)
label_array = np.array(label_img)

# === 파일 이름 추출 ===
filename = Path(label_path).stem

# === 시각화 ===
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(label_array, cmap=cmap, norm=norm)
ax.axis('off')
ax.set_title(f"OpenEarthMap - {filename}", fontsize=14)

# colorbar 추가 (이미지 세로 높이 맞춤)
cbar = plt.colorbar(im, ax=ax, ticks=np.arange(9), fraction=0.046, pad=0.04)
cbar.ax.set_yticklabels([class_info[i][0] for i in range(9)])
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
pth = f'./visualize/label/{filename}_label.png'
plt.savefig(pth, bbox_inches='tight')
