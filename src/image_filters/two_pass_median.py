import sys
import re
import os
import time

import cv2 as cv
import numpy as np

from src.math_helpers import signal_to_noise_ratio
from src.dlls import cpp_caller


def two_pass_median(n_image: cv.Mat) -> cv.Mat:
    return cpp_caller.call_two_pass_median_for_image_vector(n_image)


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
        if image is not None:
            start = time.time()
            filtered_image = two_pass_median(image)
            end = time.time()
            elapsed_time = end - start
            print(f'Elapsed time: {elapsed_time:.2f} seconds')
            cv.imshow('noisy', image)
            cv.imshow('filtered_image', filtered_image)
            # print('signal_to_noise_filtered', signal_to_noise_ratio(image, filtered_image))
            cv.waitKey()
            cv.destroyAllWindows()
            exit(0)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
