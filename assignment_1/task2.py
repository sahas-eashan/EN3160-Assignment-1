import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = "task2"
os.makedirs(save_dir, exist_ok=True)

img = cv.imread("a1images/brain_proton_density_slice.png", cv.IMREAD_GRAYSCALE)
x = np.linspace(0, 255, 256)

mu_white, sigma_white = 200, 20
lut_white = (
    (255 * np.exp(-((x - mu_white) ** 2) / (2 * sigma_white**2)))
    .clip(0, 255)
    .astype(np.uint8)
)

mu_gray, sigma_gray = 140, 20
lut_gray = (
    (255 * np.exp(-((x - mu_gray) ** 2) / (2 * sigma_gray**2)))
    .clip(0, 255)
    .astype(np.uint8)
)

white_out = cv.LUT(img, lut_white)
gray_out = cv.LUT(img, lut_gray)

for name, lut in [
    ("white_matter_curve.png", lut_white),
    ("gray_matter_curve.png", lut_gray),
]:
    plt.figure()
    plt.plot(lut)
    plt.title(name.replace("_", " ").title())
    plt.xlabel("Input intensity")
    plt.ylabel("Output intensity")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, name), dpi=300)
    plt.close()

cv.imwrite(os.path.join(save_dir, "original.png"), img)
cv.imwrite(os.path.join(save_dir, "white_matter.png"), white_out)
cv.imwrite(os.path.join(save_dir, "gray_matter.png"), gray_out)
