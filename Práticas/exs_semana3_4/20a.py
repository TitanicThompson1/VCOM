import os
import cv2 as cv
import numpy as np

imagesDir = "Images_01/"

# Opening an image
# Change this, according to your image's path
image = cv.imread(os.path.join(imagesDir, 'roundObjects_01.jpg'))
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(image, (5, 5), 0)

Z = image.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv.TermCriteria_MAX_ITER + cv.TermCriteria_EPS, 10, 1.0)
k = 2

ret, label, center = cv.kmeans(
    Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
kmeans = center[label.flatten()]
kmeans = kmeans.reshape((image.shape))


canny = cv.Canny(kmeans, 30, 150, 3)
dilated = cv.dilate(canny, (1, 1), iterations=0)

(cnt, hierarchy) = cv.findContours(
    dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

cv.drawContours(image, cnt, -1, (0, 255, 0), 2)
print(len(cnt))

cv.imshow('image', image)
cv.imshow('canny', canny)
cv.imshow('dilated', dilated)

cv.waitKey(0)
cv.destroyAllWindows()

