import numpy as np
import cv2 as cv
import os

imagesDir = '../Images_01'  # Change this, according to your images' directory path

corners_img = cv.imread(os.path.join(imagesDir, 'corners_01.jpg'))
corners_img_copy = corners_img.copy()

gray = cv.cvtColor(corners_img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

max_corners = 25
quality = 0.5
mindistance = 20

corners = cv.goodFeaturesToTrack(gray, max_corners, quality, mindistance)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv.circle(corners_img_copy, (x, y), 3, 255, -1)

cv.imshow('corners_img_copy', corners_img_copy)
cv.imshow('corners_img', corners_img)

cv.waitKey(0)
cv.destroyAllWindows()

