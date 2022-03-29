import cv2 as cv
import os

threshold = 100
aperture_size = 5

dataDir = '../Images_01'

img = cv.imread(os.path.join(dataDir, 'circles_01.jpg'), 0)
img = cv.resize(img, (719, 744))

img_canny = cv.Canny(img, threshold, threshold, apertureSize=aperture_size)

img_sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=aperture_size)
img_sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=aperture_size)

abs_sobel_x = cv.convertScaleAbs(img_sobel_x)
abs_sobel_y = cv.convertScaleAbs(img_sobel_y)

grad = cv.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
_, grad = cv.threshold(grad, threshold, 255, cv.THRESH_BINARY)

cv.imshow("Sobel", grad)
cv.imshow("Canny", img_canny)

cv.waitKey(0)

cv.destroyAllWindows()
