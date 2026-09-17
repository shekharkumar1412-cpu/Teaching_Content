import cv2, numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist(
    [gray],
    [0],
    None,
    [256],
    [0, 256]
)

plt.plot(hist)
plt.xlabel("Intensity")
plt.ylabel("Number of Pixels")
plt.show()