import os
import sys
import re
import time
import cv2 as cv

from src.math_helpers.signal_to_noise_ratio import signal_to_noise_ratio
from src.dlls import cpp_caller


def weighted_median_filter(image: cv.Mat, kernel_size: int, weight_type: str = 'uniform') -> cv.Mat:
    return cpp_caller.call_weighted_median_filter_vector(image, kernel_size, weight_type)


def convert_inputs_for_weighted_median(image_name: str, weight_type: str, kernel_size: str) -> (cv.Mat, str, int):
    """
    From the filename, gets the image and checks, and converts it to grayscale
        :param image_name: Name of the image in the 'images' folder
        :param weight_type: Type of weighting to use ('uniform' or 'distance').
        :param kernel_size: Size of the kernel, odd number, greater than 1 (3,5,7...)
        :return: Grayscale image (cv2.Mat)
    """
    __input_image = None
    w_type = ''
    kernel = 0
    try:
        pattern = re.compile(r"([\w\s]+)\.(jpg|jpeg|png)$")
        if not pattern.match(image_name):
            raise Exception('The filename must be jpg, jpeg, png!')
        if not os.path.exists(f'../../images/{image_name}'):
            raise Exception('The file does not exist!')
        __input_image = cv.imread(f'../../images/{image_name}')
        # To make sure, the input will be grayscale
        __input_image = cv.cvtColor(__input_image, cv.COLOR_BGR2GRAY)

        w_type = str(weight_type)

        assert kernel_size.isdigit(), 'the third parameter should be an integer'
        kernel = int(kernel_size)
        assert kernel % 2 == 1, 'the kernel size must be odd'
        assert kernel > 1, 'the kernel must be greater than 1'
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return __input_image, w_type, kernel


# If the program called from command line
if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 4:
        print('You must pass exactly three argument!', file=sys.stderr)
        exit(1)
    try:
        image, weight_type, kernel_s = convert_inputs_for_weighted_median(sys.argv[1], sys.argv[2], sys.argv[3])
        if image is not None:
            start = time.time()
            filtered_image = weighted_median_filter(image, kernel_s, weight_type)
            end = time.time()
            elapsed_time = end - start
            print(f'Elapsed time: {elapsed_time:.2f} seconds')
            cv.imshow('input_image', image)
            cv.imshow(f'filtered_image with {kernel_s} kernel size and {weight_type} weight type', filtered_image)
            cv.waitKey()
            cv.destroyAllWindows()
            print('signal_to_noise_filtered', signal_to_noise_ratio(image, filtered_image))
            exit(0)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)



