import cv2 as cv
import sys
import re
import os

import numpy as np
from src.math_helpers import signal_to_noise_ratio


def generate_noise_map(_apx_of_noise: cv.Mat) -> cv.Mat:
    """
    Creates noise map from a grayscale image.
    Pixels can have 1 and 0, as following:
    1, if the _apx_of_noise >= std_dev + avg_gray_level
    0, if the _apx_of_noise < std_dev + avg_gray_level
        :param _apx_of_noise: grayscale image, in cv.Mat format
        :return: the approximated noise map of the input image
    """
    avg_grey_level = np.average(_apx_of_noise)
    mean, std_dev = cv.meanStdDev(_apx_of_noise)
    noise_map = np.zeros(_apx_of_noise.shape)
    noise_map[_apx_of_noise >= std_dev + avg_grey_level] = 1
    noise_map = cv.Mat(noise_map)
    return noise_map


def two_pass_median(image_to_filter: cv.Mat) -> cv.Mat:
    """
    Two pass median filter, based on 'papers/Two_Pass_Median_Filter.pdf'.
        :param image_to_filter: Takes a grayscale image in cv2.Mat type
        :return: Filtered image in cv2.Mat format
    """
    assert len(image_to_filter.shape) == 2, 'The input image must be grayscale'
    """
    L - noisy image
    S - signal component
    N - noise component
    N'- apr. of noise
    S'- apr. of signal only
    L = S + N
    N = L - S
    N' = L - S'
    """
    noisy_image = image_to_filter.copy()
    # Median filtration, to approximate the signal-only image
    apx_of_signal_only = cv.medianBlur(image_to_filter, 3)
    apx_of_noise = abs(np.subtract(noisy_image, apx_of_signal_only))
    noise_map = generate_noise_map(apx_of_noise)
    _filtered_image = noisy_image.copy()
    _filtered_image[noise_map == 0] = apx_of_signal_only[noise_map == 0]
    return _filtered_image


def convert_inputs_for_two_pass(image_name: str) -> cv.Mat:
    """
    From the filename, gets the image and checks, if that grayscale
        :param image_name: Name of the image in the 'images' folder
        :return: Grayscale image (cv2.Mat)
    """
    __input_image = None
    try:
        pattern = re.compile(r"([\w\s]+)\.(jpg|jpeg|png)$")
        if not pattern.match(image_name):
            raise Exception('The filename must be jpg, jpeg, png!')
        if not os.path.exists(f'../../images/{image_name}'):
            raise Exception('The file does not exist!')
        __input_image = cv.imread(f'../../images/{image_name}')
        # To make sure, the input will be grayscale
        # TODO: make available for RBG pictures
        __input_image = cv.cvtColor(__input_image, cv.COLOR_BGR2GRAY)
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return __input_image


# If the program called from command line
if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 2:
        print('You must pass exactly one argument!', file=sys.stderr)
        exit(1)
    try:
        image = convert_inputs_for_two_pass(sys.argv[1])
        cv.imshow('image', image)
        if image is not None:
            filtered_image = two_pass_median(image)
            cv.imshow('filtered_image', filtered_image)
            print('signal_to_noise_filtered', signal_to_noise_ratio(image, filtered_image))
            cv.waitKey()
            cv.destroyAllWindows()
            exit(0)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
