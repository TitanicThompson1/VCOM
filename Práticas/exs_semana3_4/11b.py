import numpy as np
import cv2 as cv
import os

imagesDir = '../Images_01'  # Change this, according to your images' directory path

query = cv.imread(os.path.join(imagesDir, 'match_box01a_1.png'))
train = cv.imread(os.path.join(imagesDir, 'match_box01a_2.png'))

# Create ORB detector
orb = cv.ORB_create()

# Use ORb
# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(query, None)
kp2, des2 = sift.detectAndCompute(train, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(query, kp1, train, kp2, good,
                         None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show image
cv.imshow('BFMatcher', img3)

cv.waitKey(0)
cv.destroyAllWindows()
