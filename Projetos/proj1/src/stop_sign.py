import numpy as np
import utils
import cv2 as cv
import os
    
def export_features():
    """
    Exports the keypoints and descriptors from the stop sign reference image and to an respective file.
    """

    # Reads stop sign image
    img_object = cv.imread('dataset/reference_signs/stop_sign.jpg')

    keypoints, descriptors = sift(img_object)

    path = 'features_files/'

    # Export keypoints and descriptors of object image
    utils.export_keypoints(keypoints, os.path.join(path, 'keypoints.pkl'))
    utils.export_descriptors(descriptors, os.path.join(path, 'descriptors.pkl'))

def import_features():
    """
    Imports the respective keypoints and descriptors from a stop sign reference image.
    
    :return: An array with two elements, the keypoints and the descriptors.
    """

    path = 'features_files/'

    # Import keypoints and descriptors of object image
    keypoints = utils.import_keypoints(os.path.join(path, 'keypoints.pkl'))
    descriptors = utils.import_descriptors(os.path.join(path, 'descriptors.pkl'))
    
    return [keypoints, descriptors]

def get_matches(des_obj, des_scene):
    """
    Matching descriptor vectors from the reference image and scenery image with a FLANN based matcher.
    Get the matches that are good according Lowe's ratio test.

    :param des_obj: Descriptors from the reference image for stop signs.
    :param des_scene: Descriptors from the scenery image.

    :return: Array with good match descriptors. """

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 200)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_obj, des_scene, k=2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good_matches.append(m)

    return good_matches

def sift(image):
    """
    Detects the keypoints and computes the descriptors using SIFT detector.

    :param image: Image to apply the SIFT algorithm.

    :return: An array with two elements, the keypoints and the descriptors. """

    # Convert image to GRAY
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # Create a SIFT object
    sift = cv.SIFT_create()

    # Find keypoints and descriptors in a single step with SIFT
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return [keypoints, descriptors]
    
def find_stop_signs(image):
    """
    Finds stop signs in the image.

    :param image: The image to find stop signs in.

    :return:A list of find signs. Each sign is a dicionary with two points(pt1 and pt2, necessary to draw the rectangle) 
    and the name of the sign. """

    # Read stop sign image
    img_object = cv.imread('dataset/reference_signs/stop_sign.jpg')

    final = []

    # Import keypoints and descriptors from a stop sign reference image
    kp_obj, des_obj = import_features()

    # Get keypoints and descriptors of scenery image
    kp_scene, des_scene = sift(image)

    matches = get_matches(des_obj, des_scene)

    # Set a condition that atleast 4 matches are to be there to find the object
    if len(matches) >= 4:
        
        # Extract the locations of matched keypoints in both the images
        src_pts = np.float32([kp_obj[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        # The locations of matched keypoints in both the images are passed to find the perspective transformation
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # Verify if homography matrix is empty
        if M is None:
            return final

        h, w = img_object.shape[:2]
        pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts, M)

        rect = cv.boundingRect(np.int32(dst))
        obj_square = {
                    "pt1": (rect[0], rect[1]),
                    "pt2": (rect[0] + rect[2], rect[1] + rect[3]),
                    "name": "Stop sign"
        }

        final.append(obj_square)

    return final

if __name__ == "__main__":
    
    # Path to the images
    path = 'dataset/formatted_images/'

    image = cv.imread(os.path.join(path, 'tl_st_cw_825.png'))

    if len(find_stop_signs(image)):
        print("Stop Sign detected")
    else:
        print("Sign not detected")