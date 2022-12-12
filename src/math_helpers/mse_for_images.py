import cv2 as cv
import numpy as np


def mse_for_images(first: cv.Mat, second: cv.Mat) -> float:
    """
    Compares two grayscale images, if they represent the same picture.
    They must be grayscale, and have the same height and width
        :param first: Grayscale image, cv2.Mat object
        :param second: Grayscale image, cv2.Mat object
        :return: floating number, the MSE of the two images
    """
    assert len(first.shape) < 3 and len(second.shape) < 3, 'One of the two inputs has more than 2 dimensions'
    assert first.shape == second.shape, 'The two images must have the same size'
    height, width = first.shape
    diff = cv.subtract(first, second)
    err = np.sum(diff ** 2)
    mse = err / (float(height * width))
    return mse
