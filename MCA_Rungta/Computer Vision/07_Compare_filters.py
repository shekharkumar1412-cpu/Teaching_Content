import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mean_filtered   = cv2.blur(img, (5, 5))
gaussian_filtered = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
median_filtered = cv2.medianBlur(img, 5)

# titles = ['Original', 'Mean (Box)', 'Gaussian', 'Median']
# outputs = [img, mean_filtered, gaussian_filtered, median_filtered]

# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# for ax, title, out in zip(axes, titles, outputs):
#     ax.imshow(out)
#     ax.set_title(title)
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original ")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(mean_filtered)
plt.title("Mean (Box)")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(gaussian_filtered, cmap="gray")
plt.title("Gaussian")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(median_filtered, cmap="gray")
plt.title("median")
plt.axis("off")

plt.tight_layout()
plt.show()