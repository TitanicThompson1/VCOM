import cv2 as cv
import os

trackbar_name = "Threshold Value"
window_name = "Gradient"

def threshold_change(val):
    threshold_val = cv.getTrackbarPos(trackbar_name, window_name)
    _, out = cv.threshold(grad, threshold_val, 255, cv.THRESH_BINARY)
    cv.imshow(window_name, out)

dataDir = '../Images_01'

img = cv.imread(os.path.join(dataDir, 'chessboard_01.jpg'), 0)
img = cv.resize(img, (719, 744))


cv.imshow("Chessboard", img)

img_sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
img_sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

cv.imshow("Sobel X", img_sobel_x)
cv.imshow("Sobel Y", img_sobel_y)

cv.waitKey(0)

cv.destroyAllWindows()

abs_sobel_x = cv.convertScaleAbs(img_sobel_x)
abs_sobel_y = cv.convertScaleAbs(img_sobel_y)

grad = cv.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

cv.namedWindow(window_name)

cv.createTrackbar(trackbar_name, window_name, 0, 255, threshold_change)

threshold_change(0)

cv.waitKey(0)

cv.destroyAllWindows()

