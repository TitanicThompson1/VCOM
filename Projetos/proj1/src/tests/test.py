import unittest

import cv2 as cv
import square_blue_signs
from pathlib import Path

class BlueSquare(unittest.TestCase):
    def test_simple(self):
        image = cv.imread('dataset/simple_square_blue/yes3.png')
        if image is None:
            self.fail('Image yes3.png not loaded')
        self.assertTrue(square_blue_signs.find_square_blue_signs(image))
    
    def test_all_square_blue(self):
        
        # Read all files from directory
        path = Path('dataset/simple_square_blue')
        files = [str(x) for x in path.glob('*.png')]
        for file in files:
            image = cv.imread(file)
            if image is None:
                self.fail(f'Image {file} not loaded')

            has_bs = square_blue_signs.find_square_blue_signs(image)

            if file[:3] == "yes":
                self.assertTrue(has_bs, msg=f'on {file}')
            elif file[:2] == "no":
                self.assertFalse(has_bs, msg=f'on {file}')
