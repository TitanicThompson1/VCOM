import cv2 as cv
import preprocess


def find_square_blue_signs(image):
    """
    Finds all blue squares signs in the image

    :param image: The image to find blue squares signs in

    :return: A list of find signs. Each sign is a dicionary with two points(pt1 and pt2, necessary to draw the rectangle) 
    and the name of the sign"""

    preprocessed_image = preprocess.preprocess_image(image, True)

    squares = find_squares(preprocessed_image)

    return is_blue_square(image, squares)



def find_squares(image):
    """
    Finds all squares in the image
    
    :param image: The image to find squares in"""

    thresh_image = apply_thresholding(image)
    
    square_positions = find_square_position(thresh_image)

    return square_positions


def apply_thresholding(image):

    thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    return thresh 



def find_square_position(image):
    """
    Finds all squares position in the image

    :param image: The image to find squares in

    :return: A list of numpy arrays containing the coordinates of the squares"""
    
    MIN_SIGNAL_AREA = image.shape[0] * image.shape[1] * 0.01

    contours, _ = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    final = []

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        area = cv.contourArea(cnt)

        if area > MIN_SIGNAL_AREA and is_rectangle(approx) and cv.isContourConvex(approx):
            rectangle = cv.boundingRect(cnt)
            final.append(rectangle)

    filtered_squares = filter_squares(final)

    return filtered_squares


def is_rectangle(approx):
    return len(approx) > 3


def filter_squares(squares):
    
    areas_searched = []

    sorte_squares = sorted(squares, key=lambda x: x[2] * x[3], reverse=True)

    filtered = []
    for square in sorte_squares:
        x, y, w, h = square
        already_searched = False

        # This checks if the area was already searched
        for area in areas_searched:
            if area[0] <= x <= area[1] and area[2] <= y <= area[3]:
                already_searched = True
                break

        if already_searched:
            continue
        
        areas_searched.append((x, x + w, y, y + h))

        
        filtered.append(square)
        

    return filtered

def is_blue_square(image, squares):
    """
    Checks if the square is blue, based on the number of blue squares in the image.
    The image is considered blue if the number of blue squares is greater than a certain threshold.

    :param image: The image
    :param squares: The coordenates of the squares of the image

    :return: True if the square is blue, False otherwise"""

    res = []

    for square in squares:

        x, y, w, h = square

        square_area = w * h

        # Region of interest where potential sign is
        roi = image[y:y + h, x:x + w]

        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        
        only_blue = cv.inRange(hsv_roi, (100, 100, 100), (135, 255, 255))

        # Count blue pixels
        blue_pixels_n = cv.countNonZero(only_blue)
        
        # Percentage of blue pixels
        blue_percentage = blue_pixels_n / square_area

        if blue_percentage > 0.3:
            obj_square = {
                "pt1": (x, y),
                "pt2": (x + w, y + h),
                "name": "Blue square sign"
            }
            res.append(obj_square)

    return res


# This main function is used for testing purposes
if __name__ == "__main__":
    
    # Path to the images
    path = 'dataset/simple_medium_images/cw_tl_sp_771.png'

    image = cv.imread(path)

    if len(find_square_blue_signs(image)) > 0:
        print("Found blue square")
    else:
        print("Not found")