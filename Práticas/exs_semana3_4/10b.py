import numpy as np
import cv2 as cv
import os

imagesDir = '../Images_01'  # Change this, according to your images' directory path


new_img = cv.imread(os.path.join(imagesDir, 'stitch_building01_1.png'))
gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create()

kp = orb.detect(new_img, None)

kp, des = orb.compute(new_img, kp)

orb_img = cv.drawKeypoints(new_img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('orb_img', orb_img)
cv.imshow('new_img', new_img)

cv.waitKey(0)
cv.destroyAllWindows()
