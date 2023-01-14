import sys
import os
import re
import time

import cv2 as cv
from src.math_helpers.signal_to_noise_ratio import signal_to_noise_ratio
from src.dlls import cpp_caller


def basic_median(n_image: cv.Mat, ksize: int) -> (cv.Mat, float):
    """
    Basic median filter, it'll blur the image.
        :param n_image: Grayscale image
        :param ksize: Odd number, that is greater than 1, that will be the kernel size
        :return: Median filtered image
    """
    return cpp_caller.call_basic_median_for_image_vector(n_image, ksize)


def convert_inputs_for_basic_median(image_name: str, kernel_str: str) -> (cv.Mat, int):
    """
    Returns the image from the 'bsc_thesis\\images' folder
        :param image_name: file name from 'images' folder
        :param kernel_str: an odd number that is greater than 1
        :return: cv.Mat image object
    """
    __input_image = None
    try:
        pattern = re.compile(r"([\w\s]+)\.(jpg|jpeg|png)$")
        if not pattern.match(image_name):
            raise Exception('The filename must be jpg, jpeg, png!')
        if not os.path.exists(f'../../images/{image_name}'):
            raise Exception('The file does not exist!')
        assert kernel_str.isdigit(), 'the second parameter must be a digit!'
        assert int(kernel_str) % 2 == 1, 'the second parameter must be odd'
        assert int(kernel_str) > 1, 'the second parameter must be greater than 1'
        __input_image = cv.imread(f'../../images/{image_name}')
        # To make sure, the input will be grayscale
        __input_image = cv.cvtColor(__input_image, cv.COLOR_BGR2GRAY)
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return __input_image, int(kernel_str)


# if the program called from the command line
if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 3:
        print('You must pass exactly two argument!', file=sys.stderr)
        exit(1)
    try:
        image, kernel = convert_inputs_for_basic_median(sys.argv[1], sys.argv[2])
        cv.imshow('input_image', image)
        if image is not None:
            start = time.time()
            filtered_image, psnr = basic_median(image, kernel)
            end = time.time()
            elapsed_time = end - start
            print(f'Elapsed time: {elapsed_time:.2f} seconds')
            print('PSNR', psnr, 'db')
            # print('signal_to_noise_filtered', signal_to_noise_ratio(image, filtered_image))
            cv.imshow(f'filtered_image with {kernel} kernel size', filtered_image)
            cv.waitKey()
            cv.destroyAllWindows()
            exit(0)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
