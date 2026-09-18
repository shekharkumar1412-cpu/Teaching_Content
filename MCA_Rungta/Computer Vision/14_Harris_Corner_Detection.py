import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image1.jpeg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.title("Original Image in GrayScale")
plt.show()

gray = np.float32(gray)

# blocksize represents the neighborhood size 
# used for detecting corners.
dst = cv2.cornerHarris(
    gray,
    blockSize=2,
    ksize=3,  # ksize → size of Sobel kernel
    k=0.04
)

plt.imshow(dst, cmap="gray")
plt.title("Corner Detected")
plt.axis("off")
plt.show()
