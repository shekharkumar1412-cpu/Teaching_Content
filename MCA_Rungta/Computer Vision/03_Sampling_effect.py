import cv2
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread('sample.jpg'), cv2.COLOR_BGR2GRAY)
h, w = img.shape

factors = [1, 2, 4, 8, 16]   # down-sampling factors
fig, axes = plt.subplots(1, len(factors), figsize=(15, 4))

for ax, k in zip(axes, factors):
    small = cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_NEAREST)
    # Resize back up (nearest-neighbour) so the pixel blocks are visible at same display size
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    ax.imshow(restored, cmap='gray')
    ax.set_title(f'1/{k} resolution')
    ax.axis('off')
    
plt.tight_layout()
plt.show()

