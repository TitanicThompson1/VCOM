import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt

dataDir = '../Images_01'

img = cv.imread(os.path.join(dataDir, 'lowContrast_01.jpg'))

plt.hist(img.ravel(),256,[0,256])
plt.show()