import os
import cv2 as cv
import numpy as np

imagesDir = "Images_01/"

# Opening an image
# Change this, according to your image's path
img = cv.imread(os.path.join(imagesDir, 'roundObjects_01.jpg'))
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_gray = cv.GaussianBlur(img_gray, (5, 5), 0)

_, thresh = cv.threshold(
    img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=4)

element = cv.getStructuringElement(cv.MORPH_RECT, (1, 5) )

erosion = cv.erode(opening, element, iterations=3)

cv.imshow("opening", opening)
cv.imshow("thresh", thresh)
cv.imshow("erosion", erosion)
cv.imshow("img", img)

cv.waitKey(0)
cv.destroyAllWindows()