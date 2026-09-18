import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create black image
img = np.zeros((400, 400), dtype=np.uint8)

# Draw white rectangle
cv2.rectangle(
    img,
    (100, 100),
    (300, 300),
    255,
    3
)

plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()


gray = np.float32(img)

dst = cv2.cornerHarris(
    gray,
    blockSize=2,
    ksize=3,
    k=0.04
)

# cv2.dilate() makes bright/high-value regions slightly bigger.
dst = cv2.dilate(dst, None)

result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Applying a threshold value
threshold = 0.1* dst.max()

# marking the corner using red color
result[dst > threshold ] = [0, 0, 255]

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()