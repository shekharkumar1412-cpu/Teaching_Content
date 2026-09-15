import cv2
import matplotlib.pyplot as plt

img_bgr = cv2.imread('sample.jpg')

img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

img_cym  =  255 - img_rgb
titles = ['RGB', 'HSV', 'Gray' ,'CYM']
images = [img_rgb, img_hsv,  img_gray ,img_cym]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, title, im in zip(axes, titles, images):
    cmap = 'gray' if title == 'Gray' else None
    ax.imshow(im, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()

# Visualize individual HSV channels separately
H, S, V = cv2.split(img_hsv)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, ch, name in zip(axes, [H, S, V], ['Hue', 'Saturation', 'Value']):
    ax.imshow(ch, cmap='gray')
    ax.set_title(name)
    ax.axis('off')
plt.show()
