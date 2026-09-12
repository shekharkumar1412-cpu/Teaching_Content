# reading a digital image, display its basic properties, and view it 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image from disk (BGR order by default in OpenCV)
img = cv2.imread('sample1.jpg')

if img is None:
    raise FileNotFoundError('Check the image path')

# Basic properties -> demonstrates that an image is just a numeric array
print('Image shape (H, W, Channels):', img.shape)
print('Data type:', img.dtype)
print('Total pixels:', img.shape[0] * img.shape[1])
print('Pixel value at (0,0):', img[0, 0])

# Convert BGR (OpenCV default) to RGB for correct display with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title('Loaded Image')
plt.axis('off')
plt.show()
