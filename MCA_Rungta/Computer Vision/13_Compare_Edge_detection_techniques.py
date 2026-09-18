import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read image
img = cv2.imread("image.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel
sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,          # dx
    0,          # dy
    ksize=3
)

sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,          # dx
    1,          # dy
    ksize=3
)

# Gradient magnitude
sobel = cv2.magnitude(
    sobel_x.astype("float32"),
    sobel_y.astype("float32")
)

# Normalize for display
sobel = cv2.normalize(
    sobel,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype("uint8")


# 2. PREWITT

# Prewitt X kernel
prewitt_x_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Prewitt Y kernel
prewitt_y_kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Apply kernels
prewitt_x = cv2.filter2D(
    gray,
    cv2.CV_64F,
    prewitt_x_kernel
)

prewitt_y = cv2.filter2D(
    gray,
    cv2.CV_64F,
    prewitt_y_kernel
)

# Gradient magnitude
prewitt = np.sqrt(
    prewitt_x**2 +
    prewitt_y**2
)

# Normalize for display
prewitt = cv2.normalize(
    prewitt,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype("uint8")


# 3. CANNY
# Gaussian blur
blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    1.4
)

# Canny
canny = cv2.Canny(
    blur,
    50,
    150
)

# 4. DISPLAY

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original / Grayscale")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(prewitt, cmap="gray")
plt.title("Prewitt")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(canny, cmap="gray")
plt.title("Canny")
plt.axis("off")

plt.tight_layout()
plt.show()
