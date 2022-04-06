from __future__ import print_function
import os
import cv2 as cv
from numpy import gradient


src = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'

title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'
title_opening_window = "Opening Demo"
title_closign_window = "Closing Demo"
title_gradient_window = "Gradient Demo"
title_tophat_window = "Top-hat Demo"
title_blackhat_window = "Black-hat Demo"

def main():
    global src
    imagesDir = "Images_01/"

    src = cv.imread(os.path.join(imagesDir, 'roundObjects_03.JPG'))

    # cv.namedWindow(title_erosion_window)
    # cv.createTrackbar(title_trackbar_element_shape,
    #                   title_erosion_window, 0, max_elem, erosion)
    # cv.createTrackbar(title_trackbar_kernel_size,
    #                   title_erosion_window, 0, max_kernel_size, erosion)

    # cv.namedWindow(title_dilation_window)
    # cv.createTrackbar(title_trackbar_element_shape,
    #                   title_dilation_window, 0, max_elem, dilatation)
    # cv.createTrackbar(title_trackbar_kernel_size,
    #                   title_dilation_window, 0, max_kernel_size, dilatation)

    # cv.namedWindow(title_opening_window)
    # cv.createTrackbar(title_trackbar_element_shape,
    #                   title_opening_window, 0, max_elem, opening)
    # cv.createTrackbar(title_trackbar_kernel_size,
    #                   title_opening_window, 0, max_kernel_size, opening)

    # cv.namedWindow(title_closign_window)
    # cv.createTrackbar(title_trackbar_element_shape,
    #                   title_closign_window, 0, max_elem, closing)
    # cv.createTrackbar(title_trackbar_kernel_size,
    #                   title_closign_window, 0, max_kernel_size, closing)

    # cv.namedWindow(title_gradient_window)
    # cv.createTrackbar(title_trackbar_element_shape,
    #                   title_gradient_window, 0, max_elem, gradient)
    # cv.createTrackbar(title_trackbar_kernel_size,
    #                   title_gradient_window, 0, max_kernel_size, gradient)
                      
    # cv.namedWindow(title_tophat_window)
    # cv.createTrackbar(title_trackbar_element_shape,
    #                   title_tophat_window, 0, max_elem, tophat)
    # cv.createTrackbar(title_trackbar_kernel_size,
    #                   title_tophat_window, 0, max_kernel_size, tophat)

    cv.namedWindow(title_blackhat_window)
    cv.createTrackbar(title_trackbar_element_shape,
                      title_blackhat_window, 0, max_elem, blackhat)
    cv.createTrackbar(title_trackbar_kernel_size,
                      title_blackhat_window, 0, max_kernel_size, blackhat)

    
    # erosion(0)
    # dilatation(0)
    # opening(0)
    # closing(0)
    # gradient(0)
    # tophat(0)
    blackhat(0)

    cv.waitKey()
# optional mapping of values with morphological shapes


def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE


def erosion(val):
    element = extract_structure_form(title_erosion_window)

    erosion_dst = cv.erode(src, element)
    cv.imshow(title_erosion_window, erosion_dst)


def dilatation(val):
    element = extract_structure_form(title_dilation_window)
    dilatation_dst = cv.dilate(src, element)
    cv.imshow(title_dilation_window, dilatation_dst)


def opening(val):
    element = extract_structure_form(title_opening_window)
    dilatation_dst = cv.morphologyEx(src, cv.MORPH_OPEN, element)
    cv.imshow(title_opening_window, dilatation_dst)


def closing(val):
    element = extract_structure_form(title_closign_window)
    dilatation_dst = cv.morphologyEx(src, cv.MORPH_CLOSE, element)
    cv.imshow(title_closign_window, dilatation_dst)


def gradient(val):
    element = extract_structure_form(title_gradient_window)
    dilatation_dst = cv.morphologyEx(src, cv.MORPH_GRADIENT, element)
    cv.imshow(title_gradient_window, dilatation_dst)


def tophat(val):
    element = extract_structure_form(title_tophat_window)
    dilatation_dst = cv.morphologyEx(src, cv.MORPH_TOPHAT, element)
    cv.imshow(title_tophat_window, dilatation_dst)


def blackhat(val):
    element = extract_structure_form(title_blackhat_window)
    dilatation_dst = cv.morphologyEx(src, cv.MORPH_BLACKHAT, element)
    cv.imshow(title_blackhat_window, dilatation_dst)

def extract_structure_form(window_name):
    dilatation_size = cv.getTrackbarPos(
        title_trackbar_kernel_size, window_name)
    dilation_shape = morph_shape(cv.getTrackbarPos(
        title_trackbar_element_shape, window_name))
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
                                       
    return element


if __name__ == "__main__":
   
    main()
