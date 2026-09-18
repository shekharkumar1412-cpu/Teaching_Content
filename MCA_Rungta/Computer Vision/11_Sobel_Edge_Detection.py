import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

magnitude = cv2.magnitude(
    sobel_x.astype("float32"),
    sobel_y.astype("float32")
)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(gray, cmap="gray")
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(abs(sobel_x), cmap="gray")
plt.title("Sobel X")

plt.subplot(2,2,3)
plt.imshow(abs(sobel_y), cmap="gray")
plt.title("Sobel Y")

plt.subplot(2,2,4)
plt.imshow(magnitude, cmap="gray")
plt.title("Gradient Magnitude")

plt.show()