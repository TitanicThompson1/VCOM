import cv2 as cv
import pickle
import numpy as np


def import_image(path):
    """
    Import an image from a path.

    :param path: Path to the image.
    :return: The image.
    """
    image = cv.imread(path)
    if image is None:
        raise ValueError('Image not found.')
    return image

def show_image(image, window_name='image'):
    """
    Show an image.

    :param image: The image to show.
    """
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def export_keypoints(keypoints, path):
    """
    Export keypoints to a file.

    :param keypoints: Keypoints to export.
    :param path: Path to the file.
    """

    index = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, 
            point.class_id) 
        index.append(temp)
    f = open(path, "wb")
    f.write(pickle.dumps(index))
    f.close()

def export_descriptors(descriptors, path):
    """
    Export descriptors to a file.

    :param descriptors: descriptors to export.
    :param path: Path to the file.
    """

    f = open(path, "wb")
    np.savetxt(f, descriptors, delimiter=',', )
    f.close()

def import_keypoints(path):
    """
    Import keypoints from a file.

    :param path: Path to the file.
    :return: The keypoints.
    """

    f = open(path, "rb")
    index = pickle.loads(f.read())
    f.close()
    
    keypoints = []
    for point in index:
        temp = cv.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2], 
            response=point[3], octave=point[4], class_id=point[5])
        keypoints.append(temp)

    return tuple(keypoints)

def import_descriptors(path):
    """
    Import descriptors from a file.

    :param path: Path to the file.
    :return: The descriptors.
    """
    
    descriptors = np.loadtxt(path, delimiter=',', dtype=np.float32)

    return descriptors

def label_sign(image, signs):
    """
    Label a sign on an image.

    :param image: The image to label.
    :param signs: The signs to label.
    :param name: The name of the sign.
    :return: The labeled image.
    """
    final = image.copy()
    for sign in signs:
        cv.rectangle(final, sign["pt1"], sign["pt2"], (0, 255, 0), 3)
        cv.putText(final, sign["name"], sign["pt1"], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return final