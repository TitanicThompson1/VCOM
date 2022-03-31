import numpy as np
import cv2 as cv
import os

imagesDir = '../Images_01'  # Change this, according to your images' directory path

home_img = cv.imread(os.path.join(imagesDir, 'home.jpg'))

Z = home_img.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv.TermCriteria_MAX_ITER + cv.TermCriteria_EPS, 10, 1.0)
k = 4

ret, label, center = cv.kmeans(Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res = res.reshape((home_img.shape))

cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()
