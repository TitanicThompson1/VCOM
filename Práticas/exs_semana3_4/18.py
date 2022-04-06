import os
import cv2 as cv
import numpy as np

imagesDir = "Images_01/"

# Opening an image
# Change this, according to your image's path
image = cv.imread(os.path.join(imagesDir, 'giraffe.jpg'))


# Defining the image mask for the GrabCut output; it has the same spatial dimensions as the input image
mask = np.zeros(image.shape[:2], np.uint8)

# Define the bounding box coordinates that approximately delimitates the object of interest: (x, y, width, heigh)
rect = (10, 0, 600, 800)

# Allocating memory for the two arrays that this algorithm internally uses for the segmentation of the foreground and background
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)

# Number of iterations the algorithm will run
nIter = 5

# Applying GrabCut, using the defined bounding box
(mask, bgModel, fgModel) = cv.grabCut(image, mask,
                                       rect, bgModel, fgModel, nIter, cv.GC_INIT_WITH_RECT)

# All definite background and probable background pixels are set to 0, and all definite foreground and 
# probable foreground pixels are set to 1
outputMask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1)

# Scale the mask from the range [0, 1] to [0, 255]
outputMask = (outputMask * 255).astype("uint8")

# Apply a bitwise AND to the image using the generated mask by
# GrabCut, obtaining the final output
grabcut_output = cv.bitwise_and(image, image, mask=outputMask)

cv.imshow("GrabCut output", grabcut_output)
cv.imshow("Girrafe", image)

cv.waitKey(0)
cv.destroyAllWindows()
