import math
import cv2 as cv
import numpy as np
import os


def extract_line_points(lines, i):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    return pt1, pt2


dataDir = '../Images_01'

img_chess = cv.imread(os.path.join(dataDir, 'chessboard_02.jpg'), 0)
img_road = cv.imread(os.path.join(dataDir, 'streetLines_01.jpg'), 0)


img_chess_canny = cv.Canny(img_chess, 100, 200)
img_road_canny = cv.Canny(img_road, 100, 200)

img_chess_wlines = cv.cvtColor(img_chess_canny, cv.COLOR_GRAY2BGR)
img_road_wlines = cv.cvtColor(img_road_canny, cv.COLOR_GRAY2BGR)

lines_chess = cv.HoughLines(img_chess_canny, 1, np.pi / 180, 60)
lines_road = cv.HoughLines(img_road_canny, 1, np.pi / 180, 100)


if lines_chess is not None:
    for i in range(0, len(lines_chess)):  
        pt1, pt2 = extract_line_points(lines_chess, i)
        cv.line(img_chess_wlines, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)

if lines_road is not None:
    for i in range(0, len(lines_road)):
        pt1, pt2 = extract_line_points(lines_road, i)
        cv.line(img_road_wlines, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)


cv.imshow("Original Chess", img_chess)
cv.imshow("HoughLines Chess", img_chess_wlines)

cv.imshow("Original Road", img_road)
cv.imshow("HoughLines Road", img_road_wlines)

cv.waitKey(0)

cv.destroyAllWindows()