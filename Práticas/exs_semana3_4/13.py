import numpy as np
import cv2 as cv
import os

imagesDir = '../Images_01'  # Change this, according to your images' directory path

sudoku_img = cv.imread(os.path.join(imagesDir, 'sudoku.png'))
sudoku_img = cv.cvtColor(sudoku_img, cv.COLOR_BGR2GRAY)

_, thrs_binary_img = cv.threshold(sudoku_img, 127, 255, cv.THRESH_BINARY)
_, thrs_binary_inv_img = cv.threshold(sudoku_img, 127, 255, cv.THRESH_BINARY_INV)
_, thrs_truncate_img = cv.threshold(sudoku_img, 50, 255, cv.THRESH_TRUNC)
_, thrs_to_zero_img = cv.threshold(sudoku_img, 127, 255, cv.THRESH_TOZERO)
_, thrs_to_zero_inv_img = cv.threshold(sudoku_img, 127, 255, cv.THRESH_TOZERO_INV)

# Show images
cv.imshow('Original', sudoku_img)
cv.imshow('Binary', thrs_binary_img)
cv.imshow('Binary Inverse', thrs_binary_inv_img)
cv.imshow('Truncate', thrs_truncate_img)
cv.imshow('To Zero', thrs_to_zero_img)
cv.imshow('To Zero Inverse', thrs_to_zero_inv_img)

cv.waitKey(0)
cv.destroyAllWindows()

thrs_am_bin = cv.adaptiveThreshold(sudoku_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
# thrs_ag_bin = cv.adaptiveThreshold(sudoku_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
# thrs_am_bin2 = cv.adaptiveThreshold(sudoku_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
thrs_am_trunc = cv.adaptiveThreshold(sudoku_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)




# Show image
cv.imshow('Adaptive Mean Binary', thrs_am_bin)
# cv.imshow('Adaptive Gaussian Binary', thrs_ag_bin)
# cv.imshow('Adaptive Mean Binary 2', thrs_am_bin2)
cv.imshow('Adaptive Mean Truncate', thrs_am_trunc)

cv.waitKey(0)
cv.destroyAllWindows()