import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

save_dir = "task8"
os.makedirs(save_dir, exist_ok=True)


def zoom_cv(img, s: float, method: str = "nearest"):
    interp = cv.INTER_NEAREST if method == "nearest" else cv.INTER_LINEAR  # bilinear
    h, w = img.shape[:2]
    out = cv.resize(img, None, fx=s, fy=s, interpolation=interp)
    return out


def normalized_ssd(A, B, max_val=255.0):
    if A.shape == B.shape:
        ssd = np.sum((A - B) ** 2)
        return float(ssd / A.size)
    else:
        print("A and B must have the same shape")
        return None


up_nearest = [0] * 4
up_bilinear = [0] * 4

nssd_near = [0] * 4
nssd_blin = [0] * 4

images_big = ["im01.png", "im02.png", "taylor.jpg", "im03.png"]
images_small = ["im01small.png", "im02small.png", "taylor_small.jpg", "im03small.png"]
for i in range(4):
    small_path = "a1images/a1q5images/" + images_small[i]
    large_path = "a1images/a1q5images/" + images_big[i]

    small = cv.imread(small_path)
    large = cv.imread(large_path)

    # scale up the small by factor 4
    s = 4.0
    up_nearest[i] = zoom_cv(small, s, method="nearest")
    up_bilinear[i] = zoom_cv(small, s, method="bilinear")

    # compute normalized SSDs
    nssd_near[i] = normalized_ssd(up_nearest[i], large)
    nssd_blin[i] = normalized_ssd(up_bilinear[i], large)
    if nssd_near[i] is None or nssd_blin[i] is None:
        continue
    else:
        print(f"Normalized SSD (Nearest):  {nssd_near[i]:.6f}")
        print(f"Normalized SSD (Bilinear): {nssd_blin[i]:.6f}")

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"Nearest Neighbour (SSD: {nssd_near[i]:.5f})")
        plt.imshow(cv.cvtColor(up_nearest[i], cv.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Bilinear (SSD: {nssd_blin[i]:.5f})")
        plt.imshow(cv.cvtColor(up_bilinear[i], cv.COLOR_BGR2RGB))
        plt.axis("off")
        safe_ssd = f"{nssd_near[i]:.5f}".replace(".", "_")
        filename = f"comparison_nearest_ssd_{safe_ssd}.png"

        plt.savefig(
            os.path.join(save_dir, filename),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
