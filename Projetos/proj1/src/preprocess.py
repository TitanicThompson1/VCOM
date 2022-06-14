import cv2 as cv


def preprocess_image(image, for_blue):
    """
    Prepprocesses the image, to remove noise and make it easier to find structures.
    Applies Gaussian blurr, applies a threshold to the image and applies morphological operations to remove unwanted parts.
    Current version is biased to squares. Future versions may not be.

    :param image: The image to preprocess

    :return: The preprocessed image"""

    blurred_image = cv.bilateralFilter(image, 9, 100, 100)

    mean_shift = cv.pyrMeanShiftFiltering(blurred_image, 10, 32)

    # Only filter for blue signs (for red signs it is not needed)
    if for_blue:
        mean_shift =_filter_non_blue(mean_shift)

    gray_image = cv.cvtColor(mean_shift, cv.COLOR_BGR2GRAY)

    return gray_image


def _filter_non_blue(image):
    """
    Filters the image to only keep blue colors.

    :param image: The image to filter

    :return: The filtered image"""
    
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, (100, 100, 100), (135, 255, 255))
    res = cv.bitwise_and(image, image, mask=mask)

    return res