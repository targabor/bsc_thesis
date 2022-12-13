import unittest
import cv2 as cv
import math_helpers

import numpy as np


class TestMSEForImages(unittest.TestCase):
    def test_mse_for_images_pass(self):
        first = np.ones((200, 200))
        second = np.zeros((200, 200))
        first_i = cv.Mat(first)
        second_i = cv.Mat(second)

        assert math_helpers.mse_for_images(first_i, second_i) == 1.0

    def test_mse_for_images_pass_2(self):
        first = np.zeros((200, 200))
        second = np.ones((200, 200))
        first_i = cv.Mat(first)
        second_i = cv.Mat(second)

        assert math_helpers.mse_for_images(first_i, second_i) == 1.0

    def test_mse_for_images_same(self):
        first = np.ones((200, 200))
        second = np.ones((200, 200))
        first_i = cv.Mat(first)
        second_i = cv.Mat(second)

        assert math_helpers.mse_for_images(first_i, second_i) == 0.0
