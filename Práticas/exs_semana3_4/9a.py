import numpy as np
import cv2 as cv
import os

imagesDir = '../Images_01'  # Change this, according to your images' directory path

corners_img = cv.imread(os.path.join(imagesDir, 'corners_01.jpg'))
corners_img_copy = corners_img.copy()

gray = cv.cvtColor(corners_img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

neighbourhood = 2
aperture = 3
free_param = 0.04
dst = cv.cornerHarris(gray, neighbourhood, aperture, free_param)

dst = cv.dilate(dst, None)

threshold = 0.01
corners_img[dst > threshold * dst.max()] = [0, 0, 255]

cv.imshow('corners_img_copy', corners_img_copy)
cv.imshow('corners_img', corners_img)


cv.waitKey(0)
cv.destroyAllWindows()
