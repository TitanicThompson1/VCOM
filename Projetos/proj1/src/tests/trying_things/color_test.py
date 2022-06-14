import cv2 as cv
from cv2 import resize


image = cv.imread("color_test.jpeg")
image= cv.resize(image,(900,500))



hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

#only_blue = cv.inRange(hsv, (100, 120, 120), (135, 255, 255))
lower_red = cv.inRange(hsv, (0,70,50), (10,255,255))
upper_red = cv.inRange(hsv, (170,70,100),(180,255,255))

red_mask = lower_red + upper_red
#red_mask = cv.bitwise_or(lower_red, upper_red)

bgr = cv.cvtColor(red_mask, cv.COLOR_GRAY2BGR)

cv.imshow("image", image)
cv.imshow("bgr", cv.bitwise_and(image, bgr))
cv.waitKey(0)
cv.destroyAllWindows()