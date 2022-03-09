import numpy as np
import cv2

image = np.ones((100, 200, 3), np.uint8)

image[:, :] = (0, 255, 255)

cv2.line(image, (0, 0), (200, 100), (255, 0, 0), 5)

cv2.line(image, (0, 100), (200, 0), (0, 0, 255), 5)

cv2.imshow("paint", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
