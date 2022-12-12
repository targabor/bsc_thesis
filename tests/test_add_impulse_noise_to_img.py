import unittest

import math_helpers.mse_for_images
import src
import cv2 as cv


class TestAddImpulseNoiseToImg(unittest.TestCase):
    def test_bad_input_for_check_filename(self):
        with self.assertRaises(Exception):
            src.convert_inputs('asd.jpg', '0.125')

    def test_bad_input_for_check_percent(self):
        with self.assertRaises(Exception):
            src.convert_inputs('baboon.jpg', '1.5')

    def test_good_input(self):
        img = cv.imread('../images/baboon.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        test_img, percent = src.convert_inputs('baboon.jpg', '0.5')
        assert math_helpers.mse_for_images(img, test_img) == 0.0
        assert percent == 0.5
