import cv2 as cv
import os

imagesDir = 'Images_01'  # Change this, according to your images' directory path

# Change this, according to your image's path
butterfly = cv.imread(os.path.join(imagesDir, 'butterfly.jpg'))

segmented = cv.pyrMeanShiftFiltering(butterfly, 10, 100)
segmented2 = cv.pyrMeanShiftFiltering(butterfly, 20, 50)

cv.imshow('segmented', segmented)
cv.imshow('segmented2', segmented2)
cv.imshow('butterfly', butterfly)

cv.waitKey(0)
cv.destroyAllWindows()
