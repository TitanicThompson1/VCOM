import os
import cv2 as cv
import numpy as np

imagesDir = "Images_01/"

# Opening an image
# Change this, according to your image's path
img = cv.imread(os.path.join(imagesDir, 'roundObjects_01.jpg'))
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

background = cv.dilate(opening, kernel, iterations=3)

distTrans = cv.distanceTransform(opening, cv.DIST_L1, 5)
ret, fg = cv.threshold(distTrans, 0.7 * distTrans.max(), 255, 0)

fg = np.uint8(fg)
unknown = cv.subtract(background, fg)

# Marker labelling
ret, markers = cv.connectedComponents(fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

# Display results
cv.imshow("img", img)
cv.imshow("background", background)
cv.imshow("fg", fg)
cv.imshow("opening", opening)

cv.waitKey(0)
cv.destroyAllWindows()

