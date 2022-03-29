import cv2 as cv
import os

trackname_thrsm = "Minimum Threshold Value"
trackname_thrsM = "Maximum Threshold Value"
trackname_aperture_size = "App Value"

window_name = "Gradient"


def threshold_change(val):
    thrs_min_val = cv.getTrackbarPos(trackname_thrsm, window_name)
    thrs_max_val = cv.getTrackbarPos(trackname_thrsM, window_name)
    aperture_size = cv.getTrackbarPos(trackname_aperture_size, window_name)
    out = cv.Canny(img, thrs_min_val, thrs_max_val, apertureSize=aperture_size)
    cv.imshow(window_name, out)


dataDir = '../Images_01'

img = cv.imread(os.path.join(dataDir, 'circles_01.jpg'), 0)
img = cv.resize(img, (719, 744))

img_canny = cv.Canny(img, 100, 200, apertureSize=5)

cv.namedWindow(window_name)

cv.createTrackbar(trackname_thrsm, window_name, 0, 255, threshold_change)
cv.createTrackbar(trackname_thrsM, window_name, 0, 255, threshold_change)
cv.createTrackbar(trackname_aperture_size, window_name, 3, 7, threshold_change)

threshold_change(0)

cv.waitKey(0)

cv.destroyAllWindows()
