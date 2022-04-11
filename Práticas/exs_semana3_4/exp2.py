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

closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (7, 7)))

opened = cv.morphologyEx(closed, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (38, 38)), iterations=1)
# opened2 = cv.morphologyEx(opened, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (38, 38)), iterations=1)

opened = cv.erode(opened, cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30)), iterations=1)

n, labels, stats, centroids = cv.connectedComponentsWithStats(opened)
print(f"Number of objects: {n}")

output = img.copy()

for i in range(n):
    x = stats[i, cv.CC_STAT_LEFT]
    y = stats[i, cv.CC_STAT_TOP]
    w = stats[i, cv.CC_STAT_WIDTH]
    h = stats[i, cv.CC_STAT_HEIGHT]
    cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv.imshow("thresh", thresh)
cv.imshow("closed", closed)
cv.imshow("opened", opened)
cv.imshow("output", output)

cv.waitKey(0)
cv.destroyAllWindows()