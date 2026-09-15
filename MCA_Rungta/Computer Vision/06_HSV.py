import cv2, numpy as np

img = cv2.imread('sample.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# a pixel will be considered red if:
# 0 ≤ H ≤ 10
# 120 ≤ S ≤ 255
# 70 ≤ V ≤ 255 
lower_red1 = np.array([0, 120, 70]);   upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70]); upper_red2 = np.array([180, 255, 255])

# separating those pixels which are red based upon the range on hue scale 

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask  = cv2.bitwise_or(mask1, mask2)

# Extract the red regions
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite('red_mask.jpg', mask)
cv2.imwrite('red_segmented.jpg', result)


# separating those pixels which are green 

lower_green = np.array([35, 120, 70])
upper_green = np.array([85, 255, 255])

mask_green= cv2.inRange(hsv, lower_green, upper_green)

result_green= cv2.bitwise_and(img, img, mask=mask_green)

cv2.imwrite('greenmask.jpg', mask_green)
cv2.imwrite('greensegmented.jpg', result_green)

# separating those pixels which are blue
lower_blue = np.array([100, 120, 70])
upper_blue = np.array([140, 255, 255])

mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

result_blue = cv2.bitwise_and(img, img, mask=mask_blue)

cv2.imwrite('bluemask.jpg', mask_blue)
cv2.imwrite('bluesegmented.jpg', result_blue)