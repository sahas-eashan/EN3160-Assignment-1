import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


# Save directory
save_dir = "task7"
os.makedirs(save_dir, exist_ok=True)


einstein = cv.imread("a1images/einstein.png", cv.IMREAD_GRAYSCALE)
# assert einstein is not None

sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

print("Sobel X Filter:\n", sobel_x)
print("Sobel Y Filter:\n", sobel_y)

sobel_x_filtered = cv.filter2D(einstein, cv.CV_64F, sobel_x)
sobel_y_filtered = cv.filter2D(einstein, cv.CV_64F, sobel_y)

fig, ax = plt.subplots(1, 2, figsize=(12, 8))
ax[0].imshow(sobel_x_filtered, cmap="gray")
ax[0].set_title("Sobel X Filtered(Using filter2D)")
ax[1].imshow(sobel_y_filtered, cmap="gray")
ax[1].set_title("Sobel Y Filtered(Using filter2D)")

# save images
plt.savefig(
    os.path.join(save_dir, "sobel2D_filtered.png"), dpi=300, bbox_inches="tight"
)


def sobel_filter(img, filter):
    """Apply Sobel filter to an image."""
    rows, cols = img.shape
    filtered_img = np.zeros_like(img, dtype=np.float64)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = img[i - 1 : i + 2, j - 1 : j + 2]
            filtered_img[i, j] = np.sum(region * filter)

    return filtered_img


sobel_x_filtered_custom = sobel_filter(einstein, sobel_x)
sobel_y_filtered_custom = sobel_filter(einstein, sobel_y)

fig, ax = plt.subplots(1, 2, figsize=(12, 8))
ax[0].imshow(sobel_x_filtered_custom, cmap="gray")
ax[0].set_title("Sobel X Filtered(Using Custom Function)")
ax[1].imshow(sobel_y_filtered_custom, cmap="gray")
ax[1].set_title("Sobel Y Filtered(Using Custom Function)")

plt.savefig(
    os.path.join(save_dir, "sobel_custom_filtered.png"), dpi=300, bbox_inches="tight"
)


sobel_x_vertical = np.array([[1], [2], [1]])
sobel_x_horizontal = np.array([[1, 0, -1]])

sobel_y_vertical = np.array([[1], [0], [-1]])
sobel_y_horizontal = np.array([[1, 2, 1]])

x1 = cv.filter2D(einstein, cv.CV_64F, sobel_x_horizontal)
x2 = cv.filter2D(x1, cv.CV_64F, sobel_x_vertical)

y1 = cv.filter2D(einstein, cv.CV_64F, sobel_y_vertical)
y2 = cv.filter2D(y1, cv.CV_64F, sobel_y_horizontal)

fig, ax = plt.subplots(1, 4, figsize=(12, 8))
ax[0].imshow(x1, cmap="gray")
ax[0].set_title("Sobel X intermidiate")
ax[1].imshow(x2, cmap="gray")
ax[1].set_title("Sobel  x Final")
ax[2].imshow(y1, cmap="gray")
ax[2].set_title("Sobel Y intermidiate")
ax[3].imshow(y2, cmap="gray")
ax[3].set_title("Sobel Y Final")

plt.savefig(
    os.path.join(save_dir, "sobel_combined_filtered.png"), dpi=300, bbox_inches="tight"
)
