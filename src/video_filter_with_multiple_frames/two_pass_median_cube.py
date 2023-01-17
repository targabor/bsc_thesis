import time

import cv2 as cv
import numpy as np

import sys
import os
import re

from src.dlls import cpp_caller


def two_pass_median_cube(video_name: str, neighbors: int) -> float:
    return cpp_caller.call_two_pass_median_cube(video_name, neighbors)


def convert_args_to_parameter(video_name: str, neighbors_str: str) -> (str, int, int):
    """
    Converts the command line arguments to VideoCapture object and int for kernel size
        :param neighbors_str: specifies how many adjacent frames are considered
        :param video_name: video filename, from videos folder, must be grayscale
        :return: returns the values if they are valid
    """
    neighbors = 0
    try:
        pattern = re.compile(r"([\w\s]+)\.(mp4)$")
        if not pattern.match(video_name):
            raise Exception('The filename must be mp4')
        if not os.path.exists(f'../../videos/{video_name}'):
            raise Exception('The file does not exist!')

        assert neighbors_str.isdigit(), 'the third parameter must be a digit!'
        assert float(neighbors_str) % 1 == 0, 'the third parameter must be whole number'
        neighbors = int(neighbors_str)
        assert neighbors >= 1, 'the third parameter must be greater than 1'
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return video_name, neighbors


# if the program called from the command line
if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 3:
        print('You must pass exactly two argument!', file=sys.stderr)
        exit(1)
    try:
        noisy_video, neighbors = convert_args_to_parameter(sys.argv[1], sys.argv[2])
        start = time.time()
        psnr = two_pass_median_cube(noisy_video, neighbors)
        end = time.time()
        elapsed_time = end - start
        print('PSNR', psnr, 'db')
        print(f'Elapsed time: {elapsed_time:.2f} seconds')
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
