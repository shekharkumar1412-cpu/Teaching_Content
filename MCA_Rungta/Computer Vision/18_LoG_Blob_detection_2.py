# Create a simple image containing blobs.
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("blobs.png", 0)

scales = [1, 2 , 3 , 8 , 10]

responses = []

for sigma in scales:

    # Gaussian smoothing
    blur = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=sigma
    )

    # Laplacian
    log = cv2.Laplacian(
        blur,
        cv2.CV_64F
    )

    # Absolute response
    response = np.absolute(log)
    responses.append(response)
plt.figure(figsize=(12, 8))

for i, response in enumerate(responses):
    plt.subplot(2, 3, i + 1)
    plt.imshow(response, cmap="gray")
    plt.title(f"Sigma = {scales[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# # Maximum response across scales
# max_response = np.maximum.reduce(responses)

# # Display
# plt.imshow(max_response, cmap="gray")
# plt.title("Multi-scale LoG Response")
# plt.axis("off")
# plt.show()