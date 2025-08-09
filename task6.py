import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def my_hist_equalization(img, cdf):
    L = 256
    num_pixels = cdf[-1]
    t = np.array([(L - 1) / num_pixels * cdf[k] for k in range(L)]).astype("uint8")

    result = t[img]
    return result


# Load image
img_bgr = cv.imread("a1images/jeniffer.jpg")

# Convert to HSV and split channels
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
h, s, v = cv.split(img_hsv)

# Create mask using Saturation channel
_, mask = cv.threshold(s, 12, 255, cv.THRESH_BINARY)

# Extract foreground
foreground = cv.bitwise_and(img_bgr, img_bgr, mask=mask)

foreground_hsv = cv.cvtColor(foreground, cv.COLOR_BGR2HSV)
H_fg, S_fg, V_fg = cv.split(foreground_hsv)

hist = cv.calcHist([V_fg], [0], mask, [256], [0, 256])
x_positions = np.arange(len(hist))

cdf = hist.cumsum()

# Equalize only the foreground
v_eq = my_hist_equalization(V_fg, cdf)
hist_eq = cv.calcHist([v_eq], [0], mask, [256], [0, 256])
x_positions_eq = np.arange(len(hist_eq))

# Merge back channels
hsv_eq = cv.merge([H_fg, S_fg, v_eq])
modified_fg = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)

background = cv.bitwise_and(img_bgr, img_bgr, mask=cv.bitwise_not(mask))
final_img_bgr = cv.add(cv.cvtColor(background, cv.COLOR_BGR2RGB), modified_fg)
# Save results
save_dir = "task6"
os.makedirs(save_dir, exist_ok=True)

# 1. HSV planes subplot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(h, cmap="gray")
axes[0].set_title("Hue")
axes[0].axis("off")

axes[1].imshow(s, cmap="gray")
axes[1].set_title("Saturation")
axes[1].axis("off")

axes[2].imshow(v, cmap="gray")
axes[2].set_title("Value")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "1_hsv_planes.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. Mask
plt.imshow(mask, cmap="gray")
plt.title("Foreground Mask")
plt.axis("off")
plt.savefig(os.path.join(save_dir, "2_mask.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3. Foreground extracted
plt.imshow(cv.cvtColor(foreground, cv.COLOR_BGR2RGB))
plt.title("Extracted Foreground (Value Channel)")
plt.axis("off")
plt.savefig(os.path.join(save_dir, "3_foreground.png"), dpi=300, bbox_inches="tight")
plt.close()

# 4. Original image
plt.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")
plt.savefig(os.path.join(save_dir, "4_original.png"), dpi=300, bbox_inches="tight")
plt.close()

# 5. Final equalized image
plt.imshow(cv.cvtColor(final_img_bgr, cv.COLOR_BGR2RGB))
plt.title("Equalized Foreground")
plt.axis("off")
plt.savefig(os.path.join(save_dir, "5_result.png"), dpi=300, bbox_inches="tight")
plt.close()

# 6. Plot the histogram
plt.figure(figsize=(8, 4))
plt.bar(x_positions, hist.flatten(), color="black", width=1.0)
plt.title("Histogram of V Channel (Foreground)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.grid()
plt.savefig(os.path.join(save_dir, "6_histogram.png"), dpi=300, bbox_inches="tight")
plt.close()

# 7. Plot the CDF
plt.figure(figsize=(8, 4))
plt.plot(cdf, color="black")
plt.title("CDF of V Channel (Foreground)")
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Frequency")
plt.xlim([0, 256])
plt.grid()
plt.savefig(os.path.join(save_dir, "7_cdf.png"), dpi=300, bbox_inches="tight")
plt.close()

# 8. Plot the histogram equalized result
plt.figure(figsize=(8, 4))
plt.bar(x_positions_eq, hist_eq.flatten(), color="black", width=1.0)
plt.title("Histogram of V Channel (Foreground) - Equalized")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.grid()
plt.savefig(os.path.join(save_dir, "8_histogram_eq.png"), dpi=300, bbox_inches="tight")
plt.close()
