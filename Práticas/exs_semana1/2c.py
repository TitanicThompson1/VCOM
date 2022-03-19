import numpy as np
import cv2


image = cv2.imread("dog_print.PNG")

cv2.imshow("Normal dog", image)

grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Dog with styleee", grayscale_img)

cv2.imwrite("dog_grayscale.png", grayscale_img)

cv2.waitKey(0)

cv2.destroyAllWindows()


