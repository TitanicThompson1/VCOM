import numpy as np
import cv2 as cv
import os

# Watershed algorithm

imagesDir = '../Images_01'  # Change this, according to your images' directory path

# Change this, according to your image's path
coins = cv.imread(os.path.join(imagesDir, 'watershed_coins.jpg'))

coins_gray = cv.cvtColor(coins, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(coins_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

background = cv.dilate(opening, kernel, iterations=3)

distTrans = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, fg = cv.threshold(distTrans, 0.7 * distTrans.max(), 255, 0)

fg = np.uint8(fg)
unknown = cv.subtract(background, fg)

# Marker labelling
ret, markers = cv.connectedComponents(fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(coins, markers)
coins[markers == -1] = [255, 0, 0]

# Display results
cv.imshow("coins", coins)

cv.waitKey(0)
cv.destroyAllWindows()
