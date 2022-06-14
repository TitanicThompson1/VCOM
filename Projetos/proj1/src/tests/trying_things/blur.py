import cv2 as cv
import numpy as np


image = cv.imread('dataset/formatted_images/sp_722.png')

blurred_image = cv.medianBlur(image, 7)
# blurred_image = cv.GaussianBlur(image, (5, 5), 0)

cv.imshow("blurred_image", blurred_image)
cv.waitKey(0)
cv.destroyAllWindows()

mean_shift = cv.pyrMeanShiftFiltering(blurred_image, 10, 32)


cv.imshow("mean_shift", mean_shift)
cv.waitKey(0)
cv.destroyAllWindows()

gray = cv.cvtColor(mean_shift, cv.COLOR_BGR2GRAY)


circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=100, param2=70, minRadius=30, maxRadius=0)

circles = np.uint16(np.around(circles))
print(circles)
for i in circles[0, :]:
    center = (i[0], i[1])
    # circle center
    # cv.circle(image, center, 1, (0, 100, 100), 3)
    # circle outline
    radius = i[2]
    cv.circle(image, center, radius, (255, 0, 255), 1)

cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()