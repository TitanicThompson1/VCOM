import cv2 as cv
import numpy as np
import math

from preprocess import preprocess_image


def find_red_circles_signs(image):
    """
    Determine if a red circle sign is present in the image
    
    :param image: The image to find the red circle sign in
    """
    preprocessed = preprocess_image(image, False)

    circles = find_circles(preprocessed)
   
    return is_red_circle(image, circles)


def find_circles(image):
    """
    Finds all circles position in the image

    :param image: The image to find circles in

    :return: A list of numpy arrays containing the coordinates of the circles"""

    houghCircles= cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 2, param1=200, param2=65, minRadius=10, maxRadius=image.shape[0]//2)
    if houghCircles is None:
        return []
        
    houghCircles = np.int16(np.around(houghCircles))

    circles = []
    for i in houghCircles[0, :]:
        center = (i[0], i[1])        
        circles.append((center, i[2]))

    final_circles = filter_circles(circles)
    
    return final_circles


def filter_circles(circles):
    # Sort circles by radius
    scircles = sorted(circles, key=lambda x: x[1])

    final = []
    for i, circle in enumerate(scircles):
        center, radius = circle

        is_inside = False
        for j in range(i + 1, len(scircles)):
            if euclidian_distance(center, scircles[j][0]) < radius + scircles[j][1]:
                is_inside = True
                break
        if not is_inside:
            final.append(circle)
    
    return final

    
def euclidian_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_red_circle(image, circles):
    """
    Checks if the circle is red, based on the number of red circles in the image.
    The image is considered red if the number of red pixels is greater than a certain threshold.

    :param image: The image
    :param squares: The coordenates of the circles of the image

    :return: True if the circle is red, False otherwise"""

    final = []
    
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    for circle in circles:
        
        (x,y),r = circle
        r_float = np.float64(r)
        circle_area = math.pi*(r_float**2)

        roi = hsv_image[max(0,y-r):min(y+r, hsv_image.shape[0]),max(0,x-r):min(x+r,hsv_image.shape[1])]
    
        lower_red = cv.inRange(roi, np.array([0,50,50]), np.array([10,255,255]))
        upper_red = cv.inRange(roi, np.array([170,50,50]), np.array([179,255,255]))

        red_mask = cv.bitwise_or(lower_red, upper_red)
        
        red_pixels_n = cv.countNonZero(red_mask)
        
        # Percentage of red pixels
        red_percentage = red_pixels_n / circle_area

        if red_percentage > 0.02:
            obj_circle = {
                "pt1": (x - r, y - r),
                "pt2": (x + r, y + r),
                "name": "Red circle sign"
            }
            final.append(obj_circle)
            
    return final


# This main function is used for testing purposes
if __name__ == "__main__":
    # Path to the images
    path = 'dataset/formatted_images/occlusion5.jpeg'

    image = cv.imread(path)

    if len(find_red_circles_signs(image)) > 0:
        print("Red circle sign detected")
    else:
        print("Red circle sign not detected")
        