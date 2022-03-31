import numpy as np
import cv2 as cv
import os

imagesDir = '../Images_01'  # Change this, according to your images' directory path

query = cv.imread(os.path.join(imagesDir, 'match_box01a_1.png'))
train = cv.imread(os.path.join(imagesDir, 'match_box01a_2.png'))

orb = cv.ORB_create()

kp_query, desc_query = orb.detectAndCompute(query, None)
kp_train, desc_train = orb.detectAndCompute(train, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = bf.match(desc_query, desc_train)

matches = sorted(matches, key=lambda x: x.distance)


img_matches = cv.drawMatches(query, kp_query, train, kp_train, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

obj = np.empty((10, 2), dtype=np.float32)
scene = np.empty((10, 2), dtype=np.float32)

for i in range(10):
    obj[i, 0] = kp_query[matches[i].queryIdx].pt[0]
    obj[i, 0] = kp_query[matches[i].queryIdx].pt[1]
    scene[i, 0] = kp_train[matches[i].queryIdx].pt[0]
    scene[i, 0] = kp_train[matches[i].queryIdx].pt[1]

H, _ = cv.findHomography(obj, scene, cv.RANSAC)

obj_corners = np.empty((4, 1, 2), dtype=np.float32)
obj_corners[0, 0, 0] = 0
obj_corners[0, 0, 1] = 0
obj_corners[1, 0, 0] = query.shape[1]
obj_corners[1, 0, 1] = 0
obj_corners[2, 0, 0] = query.shape[1]
obj_corners[2, 0, 1] = query.shape[0]
obj_corners[3, 0, 0] = 0
obj_corners[3, 0, 1] = query.shape[0]

scene_corners = cv.perspectiveTransform(obj_corners, H)

cv.line(img_matches, (int(scene_corners[0, 0, 0] + query.shape[1]), int(scene_corners[0, 0, 1])),
        (int(scene_corners[1, 0, 0] + query.shape[1]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
cv.line(img_matches, (int(scene_corners[1, 0, 0] + query.shape[1]), int(scene_corners[1, 0, 1])),
        (int(scene_corners[2, 0, 0] + query.shape[1]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
cv.line(img_matches, (int(scene_corners[2, 0, 0] + query.shape[1]), int(scene_corners[2, 0, 1])),
        (int(scene_corners[3, 0, 0] + query.shape[1]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
cv.line(img_matches, (int(scene_corners[3, 0, 0] + query.shape[1]), int(scene_corners[3, 0, 1])),
        (int(scene_corners[0, 0, 0] + query.shape[1]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)

# Show the image
cv.imshow('Matches', img_matches)
cv.waitKey(0)
cv.destroyAllWindows()
