import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from pathlib import Path
import os

# === 설정 ===
LABEL_DIR = Path("./demo/label")              # 원본 mask 폴더
OUT_DIR   = Path("./visualize/label")          # 결과 저장 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 클래스 정의 및 색상 정보 (OpenEarthMap 기준) ===
class_info = {
    0: ("Background",        "#000000"),
    1: ("Bareland",          "#800000"),
    2: ("Rangeland (grass)", "#00FF24"),
    3: ("Developed space",   "#949494"),
    4: ("Road",              "#FFFFFF"),
    5: ("Tree",              "#226126"),
    6: ("Water",             "#0045FF"),
    7: ("Agriculture land",  "#4BB549"),
    8: ("Building",          "#DE1F07"),
}

# === 컬러맵·노름 생성 ===
colors = [mcolors.hex2color(class_info[i][1]) for i in range(9)]
cmap   = mcolors.ListedColormap(colors)
norm   = mcolors.BoundaryNorm(np.arange(10) - 0.5, cmap.N)

# === 모든 tif 파일에 대해 처리 ===
for label_path in sorted(LABEL_DIR.glob("*.tif")):
    filename = label_path.stem
    print(f"Processing {filename}…")

    # -- mask 불러오기 --
    label_img   = Image.open(label_path)
    label_array = np.array(label_img)

    # -- 시각화 --
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(label_array, cmap=cmap, norm=norm)
    ax.axis('off')
    ax.set_title(f"OpenEarthMap – {filename}", fontsize=14)

    # -- colorbar 추가 --
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(9),
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([class_info[i][0] for i in range(9)])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / f"{filename}_label.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

print("All done!")  
