import os
import cv2 as cv
import numpy as np


def Hist_and_Backproj(val):

    bins = val
    histSize = max(bins, 2)
    ranges = [0, 180]  # hue_range

    hist = cv.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    backproj = cv.calcBackProject([hue], [0], hist, ranges, scale=1)

    cv.imshow('BackProj', backproj)

    w = 400
    h = 400
    bin_w = int(round(w / histSize))
    histImg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(bins):
        cv.rectangle(histImg, (i*bin_w, h), ((i+1)*bin_w, h -
                     int(np.round(hist[i]*h/255.0))), (0, 0, 255), cv.FILLED)
    cv.imshow('Histogram', histImg)


imagesDir = 'Images_01'  # Change this, according to your images' directory path

src = cv.imread(os.path.join(imagesDir, 'hist_b-p_sky_grass_01.jpg'))


hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

# For this tutorial, we will use only the Hue value for our 1-D histogram 
# (check out the fancier code in the links above if you want to use the more standard H-S histogram, which yields better results):
ch = (0, 0)
hue = np.empty(hsv.shape, hsv.dtype)
cv.mixChannels([hsv], [hue], ch)

window_image = 'Source image'
cv.namedWindow(window_image)
bins = 25
cv.createTrackbar('* Hue  bins: ', window_image, bins, 180, Hist_and_Backproj)

Hist_and_Backproj(bins)

cv.imshow(window_image, src)
cv.waitKey()
