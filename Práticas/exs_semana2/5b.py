import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

dataDir = '../Images_01'

img = cv.imread(os.path.join(dataDir, 'lowContrast_01.jpg'), 0)

cv.imshow("Normal", img)

# b1)
img_equilized = cv.equalizeHist(img)

# b2)
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# img_equilized = clahe.apply(img)

cv.imshow("Equilized", img_equilized)

cv.waitKey(0)

cv.destroyAllWindows()
