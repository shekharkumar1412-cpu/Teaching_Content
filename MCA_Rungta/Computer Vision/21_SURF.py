import cv2

img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()

keypoints, descriptors = surf.detectAndCompute(
    gray, None
)

print("Number of keypoints:", len(keypoints))
print("Descriptor shape:", descriptors.shape)
