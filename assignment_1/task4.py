import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = "task4"
os.makedirs(save_dir, exist_ok=True)

img = cv.imread("a1images/spider.png")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
H, S, V = cv.split(hsv)

a = 0.65
sigma = 70.0
x = np.arange(256, dtype=np.float32)
f = x + a * 128.0 * np.exp(-((x - 128.0) ** 2) / (2.0 * sigma**2))
lut = np.clip(f, 0, 255).astype(np.uint8)

S_enh = cv.LUT(S, lut)
hsv_enh = cv.merge((H, S_enh, V))
img_enh = cv.cvtColor(hsv_enh, cv.COLOR_HSV2BGR)

plt.figure(figsize=(5, 4))
plt.plot(x, lut)
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.xlabel("Input intensity")
plt.ylabel("Output intensity")
plt.title(f"Vibrance transform (a={a}, σ={sigma})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "transform_curve.png"), dpi=300)
plt.close()

cv.imwrite(os.path.join(save_dir, "original.png"), img)
cv.imwrite(os.path.join(save_dir, "enhanced.png"), img_enh)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(H, cmap="gray", vmin=0, vmax=255)
plt.title("Hue")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(S, cmap="gray", vmin=0, vmax=255)
plt.title("Saturation")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(V, cmap="gray", vmin=0, vmax=255)
plt.title("Value")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "hsv_planes.png"), dpi=300)
plt.close()

print("a =", a)
