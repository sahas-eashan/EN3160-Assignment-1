import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = "task3"
os.makedirs(save_dir, exist_ok=True)

img = cv.imread("a1images/highlights_and_shadows.jpg")
lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
L, a, b = cv.split(lab)

gamma = 0.5
L_corrected = np.clip((L / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
lab_corrected = cv.merge((L_corrected, a, b))
img_corrected = cv.cvtColor(lab_corrected, cv.COLOR_Lab2BGR)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Original")
plt.savefig(os.path.join(save_dir, "original_image.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.imshow(cv.cvtColor(img_corrected, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Gamma Corrected (γ={gamma})")
plt.savefig(
    os.path.join(save_dir, "gamma_corrected_image.png"), dpi=300, bbox_inches="tight"
)
plt.close()

plt.hist(
    L.ravel(), bins=256, range=(0, 255), color="black", alpha=0.6, label="Original L"
)
plt.hist(
    L_corrected.ravel(),
    bins=256,
    range=(0, 255),
    color="orange",
    alpha=0.6,
    label="Gamma Corrected L",
)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Histograms of the L plane")
plt.legend()
plt.savefig(os.path.join(save_dir, "L_plane_histogram.png"), dpi=300)
plt.close()

colors = ("b", "g", "r")
for i, col in enumerate(colors):
    plt.hist(
        img[:, :, i].ravel(),
        bins=256,
        range=(0, 255),
        color=col,
        alpha=0.5,
        label=f"Original {col.upper()}",
    )
    plt.hist(
        img_corrected[:, :, i].ravel(),
        bins=256,
        range=(0, 255),
        color=col,
        alpha=0.2,
        linestyle="dashed",
        label=f"Corrected {col.upper()}",
    )
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Histograms after converting to RGB")
plt.legend()
plt.savefig(os.path.join(save_dir, "RGB_histogram.png"), dpi=300)
plt.close()
