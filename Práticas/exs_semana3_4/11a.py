import numpy as np
import cv2 as cv
import os

imagesDir = '../Images_01'  # Change this, according to your images' directory path

query = cv.imread(os.path.join(imagesDir, 'match_box01a_1.png'))
train = cv.imread(os.path.join(imagesDir, 'match_box01a_2.png'))

# Show the images
cv.imshow('Query', query)
cv.imshow('Train', train)

# Create ORB detector
orb = cv.ORB_create()

# Use ORb
kp1, des1 = orb.detectAndCompute(query, None)
kp2, des2 = orb.detectAndCompute(train, None)

# Create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
match_output = cv.drawMatches(
    query, kp1, train, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show match
cv.imshow('Matches', match_output)

cv.waitKey(0)
cv.destroyAllWindows()
