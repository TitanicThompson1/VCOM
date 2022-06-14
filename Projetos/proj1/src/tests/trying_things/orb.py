import numpy as np
import utils
import cv2 as cv

img = cv.imread('stop_sign.jpg')
img1 = cv.imread('st_831.png')

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

kp1 = orb.detect(img1, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

kp1, des1 = orb.compute(img1, kp1)

# utils.export_keypoints(kp, 'orb_keypoints.pkl')

utils.export_descriptors(des, 'orb_keypoints.pkl')

# kp2 = utils.import_keypoints('orb_keypoints.pkl')

# img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# img3 = cv.drawKeypoints(img, kp2, None, color=(0,255,0), flags=0)

# BRUTE FORCE

# create BFMatcher object for OBR
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# MATCH()
# match descriptors with MATCH() (returns the best match)
matches = bf.match(des, des1)

# sort them in the order of their distance.
matches = sorted(matches, key=lambda x:x.distance)

# draw first 10 matches.
img_result = cv.drawMatches(img, kp, img1, kp1, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # KNNMATCH()
# # match descriptors with KNNMATCH() (returns k best matches)
# matches = bf.knnMatch(des, des1, k=2)

# # apply ratio test
# good_matches = []
# for m,n in matches:
#     if m.distance < 0.8*n.distance:
#         good_matches.append([m])

# # cv.drawMatchesKnn expects list of lists as matches.
# img_result = cv.drawMatchesKnn(img, kp, img1, kp1, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow("Matches result", img_result)

cv.waitKey(0)
cv.destroyAllWindows()