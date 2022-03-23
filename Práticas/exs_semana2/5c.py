import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

dataDir = '../Images_01'

img = cv.imread(os.path.join(dataDir, 'lowContrast_05.jpg'))

# blue, green, red = cv.split(img)

# plt.hist(blue.ravel(), 256, [0, 256], color="blue")
# plt.hist(green.ravel(), 256, [0, 256], color="green")
# plt.hist(red.ravel(), 256, [0, 256], color="red")
# plt.show()

cv.imshow("Normal", img)

# b1)
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hue, sat, val = cv.split(hsv_img)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # b2
# sat_eq = cv.equalizeHist(sat)
sat_eq = clahe.apply(sat)

# val_eq = cv.equalizeHist(val)
val_eq = clahe.apply(val)

equilazed_img = cv.cvtColor(cv.merge((hue, sat_eq, val_eq)), cv.COLOR_HSV2BGR)

cv.imshow("Equilized", equilazed_img)

cv.waitKey(0)

cv.destroyAllWindows()

