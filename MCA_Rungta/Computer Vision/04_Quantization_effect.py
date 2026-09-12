import cv2, numpy as np
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread('sample.jpg'), cv2.COLOR_BGR2GRAY)

def quantize(img, levels):
    step = 256 // levels
    return (img // step) * step

bit_levels = [256, 16, 8, 4, 2]   # corresponds to 8,4,3,2,1 bits
fig, axes = plt.subplots(1, len(bit_levels), figsize=(15, 4))

for ax, L in zip(axes, bit_levels):
    q = quantize(img, L)
    ax.imshow(q, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'{L} levels')
    ax.axis('off')

plt.tight_layout()
plt.show()
