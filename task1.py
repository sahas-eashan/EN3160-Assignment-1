import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = "task1"
os.makedirs(save_dir, exist_ok=True)


c = np.array([(50, 50), (150, 255)], dtype=np.uint8)
t1 = np.linspace(0, 50, 51, dtype=np.uint8)
t2 = np.linspace(100, 255, 150 - 50, dtype=np.uint8)
t3 = np.linspace(150, 255, 256 - 150, dtype=np.uint8)

lut = np.concatenate((t1, t2, t3), axis=0)
lut = lut[:256]

# Load grayscale image
img = cv.imread("a1images/emma.jpg", cv.IMREAD_GRAYSCALE)

# Apply LUT
out = cv.LUT(img, lut)

# Plot the transformation curve
plt.figure()
plt.plot(lut)
plt.title("Intensity Transformation")
plt.xlabel("Input intensity")
plt.ylabel("Output intensity")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "transformation_curve.png"), dpi=300)
plt.show()

# Show images
cv.imshow("Original", img)
cv.imshow("Transformed", out)
cv.imwrite(os.path.join(save_dir, "original.png"), img)
cv.imwrite(os.path.join(save_dir, "transformed.png"), out)
cv.waitKey(0)
cv.destroyAllWindows()
