import numpy as np
import cv2 as cv
import os

imagesDir = '../Images_01'  # Change this, according to your images' directory path

corners_img = cv.imread(os.path.join(imagesDir, 'corners_01.jpg'))
corners_img = cv.resize(corners_img, (0, 0), fx=0.25, fy=0.25)

fast = cv.FastFeatureDetector_create()

kp = fast.detect(corners_img, None)

corners_detected = cv.drawKeypoints(corners_img, kp, None, color=(0, 0, 255))

# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

cv.imshow('corners_img_copy', corners_img)
cv.imshow('corners', corners_detected)

cv.waitKey(0)
cv.destroyAllWindows()
