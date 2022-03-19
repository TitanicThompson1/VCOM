import numpy as np
import cv2

image = cv2.imread("dog_print.PNG")
cv2.imshow("Dog", image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

hue_channel, saturation_channel, value_channel = cv2.split(image)

cv2.imshow("Hue channel", hue_channel)
cv2.imshow("Saturation channel", saturation_channel)
cv2.imshow("Value channel", value_channel)

cv2.waitKey(0)

cv2.destroyAllWindows()

img_merged = cv2.merge((hue_channel + 10, saturation_channel + 10, value_channel + 10))

cv2.imshow("Merged Image", img_merged)

cv2.waitKey(0)

cv2.destroyAllWindows()
