import cv2, numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Histogram Equalization ---
equalized = cv2.equalizeHist(gray)

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
for ax, title, out in zip(axes, ['Original', 'Hist. Equalized'],
                           [gray, equalized]):
    ax.imshow(out, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()

# Plot histograms before/after equalization to observe redistribution
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.hist(gray.ravel(), 256, [0,256]); plt.title('Original Histogram')
plt.subplot(1,2,2); plt.hist(equalized.ravel(), 256, [0,256]); plt.title('Equalized Histogram')
plt.show()

