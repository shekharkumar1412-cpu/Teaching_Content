import numpy as np
import matplotlib.pyplot as plt
import cv2
3
# generating a image 
image = np.array([
    [0,   0,   0,   0,   0],
    [0, 255, 255, 255, 0],
    [0, 255,   0, 255, 0],
    [0, 255, 255, 255, 0],
    [0,   0,   0,   0,   0]
])

plt.imshow(image, cmap="gray")
plt.show()


image1 = np.array([
    [100,  250,   10,  200,   90],
    [50, 255, 255, 255, 0],
    [40, 255,   0, 255, 0],
    [30, 255, 255, 255, 0],
    [20,   10,   10,   50,   90]
])

plt.imshow(image1, cmap="gray")
plt.show()


# generating a random color image 

image2 = np.array([
    [[255, 0, 0],  [0, 255, 0],   [0, 0, 255], [255, 255, 0]],
    [[0, 0, 0] , [255, 255, 255], [0, 0, 0], [255, 255, 255]],
    [[0, 0, 0] , [255, 255, 255], [0, 0, 0], [255, 255, 255]],
    [[255, 0, 0],  [0, 255, 0],   [0, 0, 255], [255, 255, 0]]
])

plt.imshow(image2, cmap="gray")
plt.show()


# =======================================================================

import numpy as np
import matplotlib.pyplot as plt

# Image size
height = 500
width = 500

# Create a black image
image = np.zeros((height, width,3), dtype=np.uint8)

# Circle parameters
center_x = 250
center_y = 250
radius = 100

# Visit every pixel
for y in range(height):
    for x in range(width):

        # Circle equation:
        # (x - center_x)^2 + (y - center_y)^2 <= radius^2

        if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
            image[y, x] = [55,100,150]
        else:
            image[y, x] = [20, 155, 100]


# Display the image
plt.imshow(image)
plt.axis("off")
plt.show()