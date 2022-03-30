import cv2 as cv
import numpy as np
import os

dataDir = '../Images_01'

# coins_img = cv.imread(os.path.join(dataDir, 'coins_01.jpg'), 0)
coins_img = cv.imread(os.path.join(dataDir, 'coins_02.jpg'), 0)

coins_img = cv.blur(coins_img, (3, 3))

# Copy edges to the image that will display the results
coinsCopy = cv.cvtColor(coins_img, cv.COLOR_GRAY2RGB)

# Apply the Hough circle transform
detectionMethod = cv.HOUGH_GRADIENT                 # corresponds to the canny filter
resolutionFlag = 1                                  # same resolution as the original image
minDistance = 20                                    # between the centers of the detected circles

# param1 and param2 are the thresholds passed to the detection method 
circles = cv.HoughCircles(coins_img, detectionMethod, resolutionFlag,
                          minDistance, param1=200, param2=180, minRadius=10, maxRadius=0)
circles = np.uint16(np.around(circles))

# Drawing the resulting circles
for i in circles[0,:]: 
    cv.circle(coinsCopy,(i[0],i[1]),i[2],(0,255,0),2)

cv.imshow("Coins", coins_img)
cv.imshow("Coin lines", coinsCopy)

cv.waitKey(0)

cv.destroyAllWindows()
