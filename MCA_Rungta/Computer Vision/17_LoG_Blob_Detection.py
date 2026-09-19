# Create a simple image containing blobs.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create black image
img = np.zeros((400, 400), dtype=np.uint8)

# Draw blobs
cv2.circle(img, (100, 100), 20, 255, -1)
cv2.circle(img, (250, 120), 40, 255, -1)
cv2.circle(img, (150, 280), 60, 255, -1)

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Original Image")
plt.show()

scales = [1, 3, 5, 10, 20, 30, 40, 50]

responses = []

for sigma in scales:

    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=sigma
    )

    log = cv2.Laplacian(
        blur,
        cv2.CV_64F
    )

    response = np.absolute(log)

    responses.append(response)

plt.figure(figsize=(15, 8))

for i, response in enumerate(responses):

    plt.subplot(2, 4, i + 1)

    plt.imshow(response, cmap="gray")
    plt.title(f"Sigma = {scales[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()