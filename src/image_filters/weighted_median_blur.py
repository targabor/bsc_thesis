import os
import sys
import re

import cv2 as cv
import numpy as np

from src.math_helpers.signal_to_noise_ratio import signal_to_noise_ratio


def weighted_median_filter(image: cv.Mat, kernel_size: int, weight_type: str = 'uniform') -> cv.Mat:
    """
    Apply a weighted median filter to the input image.
        :param: image: Input image (grayscale or color).
        :param: kernel_size: Size of the median filter kernel.
        :param: weight_type: Type of weighting to use ('uniform' or 'distance').
        :return:
    """
    padding = kernel_size // 2
    padded_image = cv.copyMakeBorder(image, padding, padding, padding, padding, cv.BORDER_REPLICATE)

    output_image = np.empty_like(image)

    if weight_type == 'uniform':
        weights = np.ones((kernel_size, kernel_size))
    elif weight_type == 'distance':
        center = (kernel_size - 1) // 2
        x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
        distances = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        weights = np.exp(-distances / kernel_size)

        print(weights)
    else:
        raise ValueError("Invalid weight type. Use 'uniform' or 'distance'.")

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            neighborhood = padded_image[y:y + kernel_size, x:x + kernel_size]
            weighted_values = [neighborhood * weights]
            stacked_values = np.stack(weighted_values, axis=2)
            flattened_values = stacked_values.flatten()
            sorted_values = np.sort(flattened_values)
            median = sorted_values[len(sorted_values) // 2]
            output_image[y, x] = median

    return cv.Mat(output_image.astype(np.uint8))


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
        cv.imshow('input_image', image)
        if image is not None:
            filtered_image = weighted_median_filter(image, kernel_s, weight_type)
            cv.imshow(f'filtered_image with {kernel_s} kernel size and {weight_type} weight type', filtered_image)
            print('signal_to_noise_filtered', signal_to_noise_ratio(image, filtered_image))
            cv.waitKey()
            cv.destroyAllWindows()
            exit(0)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)



