import time

import cv2 as cv
import numpy as np

import sys
import os
import re

from src.dlls import cpp_caller


def weighted_median_cube(video_name: str, kernel_size: int, weight_type: str, neighbors: int) -> float:
    return cpp_caller.call_weighted_median_cube(video_name, kernel_size, weight_type, neighbors)


def convert_args_to_parameter(video_name: str, kernel_str: str, weight_str: str, neighbors_str: str) -> (str, int, str, int):
    """
    Converts the command line arguments to VideoCapture object and int for kernel size
        :param weight_str: specifies the type of weight
        :param neighbors_str: specifies how many adjacent frames are considered
        :param video_name: video filename, from videos folder, must be grayscale
        :param kernel_str: kernel size, must be odd number, and greater than 1 (3,5,7, etc.)
        :return: returns the values if they are valid
    """
    kernel_size = 0
    neighbors = 0
    try:
        pattern = re.compile(r"([\w\s]+)\.(mp4)$")
        if not pattern.match(video_name):
            raise Exception('The filename must be mp4')
        if not os.path.exists(f'../../videos/{video_name}'):
            raise Exception('The file does not exist!')
        assert kernel_str.isdigit(), 'the second parameter must be a digit!'
        assert float(kernel_str) % 1 == 0, 'the second parameter must be whole number'
        kernel_size = int(kernel_str)
        assert kernel_size % 2 == 1, 'the second parameter must be odd'
        assert kernel_size > 1, 'the second parameter must be greater than 1'

        assert neighbors_str.isdigit(), 'the fourth parameter must be a digit!'
        assert float(neighbors_str) % 1 == 0, 'the fourth parameter must be whole number'
        neighbors = int(neighbors_str)
        assert neighbors >= 1, 'the fourth parameter must be greater than 1'
        assert weight_str in ('uniform', 'distance'), 'the third parameter must be "uniform" or "distance"'
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return video_name, kernel_size, weight_str, neighbors


# if the program called from the command line
if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 4:
        print('You must pass exactly three argument!', file=sys.stderr)
        exit(1)
    try:
        noisy_video, kernel, weight_t, neighbors = convert_args_to_parameter(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        start = time.time()
        psnr = weighted_median_cube(noisy_video, kernel, weight_t, neighbors)
        end = time.time()
        elapsed_time = end - start
        print('PSNR', psnr, 'db')
        print(f'Elapsed time: {elapsed_time:.2f} seconds')
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
