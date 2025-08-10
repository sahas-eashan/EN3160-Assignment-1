import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


save_dir = "task9"
os.makedirs(save_dir, exist_ok=True)
# Load the image
img = cv.imread("a1images/daisy.jpg")

# GrabCut segmentation
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 100, 550, 550)

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# Convert mask to binary foreground/background
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
foreground = img * mask2[:, :, np.newaxis]
background = img * (1 - mask2[:, :, np.newaxis])

# Blur background
blurred_bg = cv.GaussianBlur(background, (51, 51), 0)
enhanced_img = cv.add(foreground, blurred_bg)

# Convert mask to 3-channel for visualization
mask_vis = cv.cvtColor(mask2 * 255, cv.COLOR_GRAY2BGR)

# Stack all in 1 row
row_img = np.hstack([mask_vis, foreground, background, img, enhanced_img])

# Save as single image
cv.imwrite(os.path.join(save_dir, "q9_outputs.png"), row_img)


plt.figure(figsize=(20, 5))
plt.imshow(cv.cvtColor(row_img, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
