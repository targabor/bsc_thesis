import sys
import re
import os
import time

import cv2 as cv
import numpy as np

from src.math_helpers.signal_to_noise_ratio import signal_to_noise_ratio
from cpp_calculate import directional_weighted_median

def convert_inputs_for_weighted(image_name: str) -> cv.Mat:
    """
    From the filename, gets the image and checks, and converts it to grayscale
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
    sys.setrecursionlimit(2000)
    input_image = None
    if len(sys.argv) != 3:
        print('You must pass exactly two argument!', file=sys.stderr)
        exit(1)
    assert sys.argv[2].isdigit(), 'The second parameter should be an integer'

    try:
        image = convert_inputs_for_weighted(sys.argv[1])
        if image is not None:
            t = int(sys.argv[2])
            start = time.time()
            filtered_image = directional_weighted_median(image, t, image.shape[0], image.shape[1])
            filtered_image = cv.Mat(np.array(filtered_image).astype(np.uint8))
            end = time.time()
            elapsed_time = end - start
            print(f'Elapsed time: {elapsed_time:.2f} seconds')
            cv.imshow('input_image', image)
            cv.imshow(f'filtered_image with {t} threshold', filtered_image)
            print('signal_to_noise_filtered', signal_to_noise_ratio(image, filtered_image))
            cv.waitKey()
            cv.destroyAllWindows()
            exit(0)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
