import cv2
import matplotlib.pyplot as plt
img = cv2.imread("image.jpg")

alpha = 1.5
beta = 30

enhanced = cv2.convertScaleAbs(
    img,
    alpha=alpha,
    beta=beta
)
cv2.imshow("Originial",img)
plt.show()
cv2.imshow("Enhanced", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()