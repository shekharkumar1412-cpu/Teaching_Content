import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image1.jpeg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Implementing Gaussian blur to remove noise 
gray = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

# Implementing Shi-Tomasi Edge Detection
corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=100, # Maximum number of corners you want.
    qualityLevel=0.01, # Lower value: more no of corners
    minDistance=10     # Minimum distance between detected corners.
)

corners = np.int32(corners)

for corner in corners:
    x, y = corner.ravel()

    cv2.circle(
        image,
        (x, y),
        5,
        (0, 0, 255),
        -1
    )

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis("off")
plt.show()