import cv2 as cv
import os

imagesDir = 'Images_01'  # Change this, according to your images' directory path

# Change this, according to your image's path
butterfly = cv.imread(os.path.join(imagesDir, 'hist_b-p_sky_grass_02.jpg'))

segmented = cv.pyrMeanShiftFiltering(butterfly, 10, 30)
segmented2 = cv.pyrMeanShiftFiltering(butterfly, 10, 60)
segmented3 = cv.pyrMeanShiftFiltering(butterfly, 20, 60)

cv.imshow('segmented', segmented)
cv.imshow('segmented2', segmented2)
cv.imshow('segmented3', segmented3)
cv.imshow('butterfly', butterfly)

cv.waitKey(0)
cv.destroyAllWindows()
