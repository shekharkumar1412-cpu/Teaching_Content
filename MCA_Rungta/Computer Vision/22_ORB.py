import cv2
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(
    image,
    None
)

print("Number of keypoints:", len(keypoints))
print("Descriptor shape:", descriptors.shape)


# Visualize ORB
output = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    color=None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.imshow(output, cmap="gray")
plt.axis("off")
plt.show()