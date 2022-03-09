import numpy as np
import cv2

image = cv2.imread("dog_print.PNG")

blue_img, green_img, red_img = cv2.split(image)

cv2.imshow("Dog", image)
cv2.imshow("Blue channel", blue_img)
cv2.imshow("Green channel", green_img)
cv2.imshow("Red channel", red_img)

cv2.waitKey(0)

cv2.destroyAllWindows()

img_merged = cv2.merge((blue_img + 10, green_img + 10, red_img + 10))

cv2.imshow("Merged Image", img_merged)

cv2.waitKey(0)

cv2.destroyAllWindows()
