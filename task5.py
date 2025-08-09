import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def my_hist_equalization(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (
        (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    )
    cdf_final = np.ma.filled(cdf_masked, 0).astype("uint8")
    return cdf_final[img]


# Load grayscale image
img = cv.imread("a1images/shells.tif", cv.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Check file path.")

# Apply histogram equalization
img_eq = my_hist_equalization(img)

# Save directory
save_dir = "task5"
os.makedirs(save_dir, exist_ok=True)

# Save original image
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.savefig(
    os.path.join(save_dir, "1_original_image.png"), dpi=300, bbox_inches="tight"
)
plt.close()

# Save equalized image
plt.imshow(img_eq, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")
plt.savefig(
    os.path.join(save_dir, "2_equalized_image.png"), dpi=300, bbox_inches="tight"
)
plt.close()

# Save original histogram
plt.hist(img.ravel(), bins=256, range=(0, 255), color="black")
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.savefig(
    os.path.join(save_dir, "3_original_histogram.png"), dpi=300, bbox_inches="tight"
)
plt.close()

# Save equalized histogram
plt.hist(img_eq.ravel(), bins=256, range=(0, 255), color="black")
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.savefig(
    os.path.join(save_dir, "4_equalized_histogram.png"), dpi=300, bbox_inches="tight"
)
plt.close()
