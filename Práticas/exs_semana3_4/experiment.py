import os
import cv2 as cv
import numpy as np

imagesDir = "Images_01/"

# Opening an image
# Change this, according to your image's path
img = cv.imread(os.path.join(imagesDir, 'roundObjects_01.jpg'))
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_gray = cv.medianBlur(img_gray, 5)

_, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# kernel = np.ones((3, 3), np.uint8)
# opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=4)

circles = cv.HoughCircles(thresh, cv.HOUGH_GRADIENT, 1, img_gray.shape[0] / 8,
                            param1=100, param2=30, minRadius=1, maxRadius=30)

if circles is not None:
    print("Number of objects: " + str(len(circles[0])))

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv.circle(img, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(img, center, radius, (255, 0, 255), 3)

# cv.imshow("opening", opening)
# cv.imshow("thresh", thresh)
cv.imshow("img", img)

cv.waitKey(0)
cv.destroyAllWindows()