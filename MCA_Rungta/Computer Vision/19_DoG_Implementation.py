import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("blobs.png", 0)

# Gaussian blur with two different sigma values
g1 = cv2.GaussianBlur(img, (0, 0), 1)
g2 = cv2.GaussianBlur(img, (0, 0), 2)

# Difference of Gaussians
dog = g2.astype(np.float64) - g1.astype(np.float64)

# Absolute value for visualization
dog = np.absolute(dog)

# Normalize to 0-255
dog = cv2.normalize(
    dog,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

dog = np.uint8(dog)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(dog, cmap="gray")
plt.title("DoG")
plt.axis("off")

plt.show()