import time

import cv2 as cv
import numpy as np

import sys
import os
import re

from src.dlls import cpp_caller


def dir_w_median_cube(video_name: str, threshold: int, neighbors: int) -> float:
    return cpp_caller.call_dir_w_median_cube(video_name, threshold, neighbors)


def convert_args_to_parameter(video_name: str, threshold_str: str, neighbors_str: str) -> (str, int, int):
    """
    Converts the command line arguments to VideoCapture object and int for kernel size
        :param neighbors_str: specifies how many adjacent frames are considered
        :param video_name: video filename, from videos folder, must be grayscale
        :param threshold_str: threshold, must be whole number, and greater than 0
        :return: returns the values if they are valid
    """
    threshold = 0
    neighbors = 0
    try:
        pattern = re.compile(r"([\w\s]+)\.(mp4)$")
        if not pattern.match(video_name):
            raise Exception('The filename must be mp4')
        if not os.path.exists(f'../../videos/{video_name}'):
            raise Exception('The file does not exist!')
        assert threshold_str.isdigit(), 'the second parameter must be a digit!'
        assert float(threshold_str) % 1 == 0, 'the second parameter must be whole number'
        threshold = int(threshold_str)
        assert threshold > 0, 'the second parameter must be greater than 0'

        assert neighbors_str.isdigit(), 'the third parameter must be a digit!'
        assert float(neighbors_str) % 1 == 0, 'the third parameter must be whole number'
        neighbors = int(neighbors_str)
        assert neighbors >= 1, 'the third parameter must be greater than 1'
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return video_name, threshold, neighbors


# if the program called from the command line
if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 4:
        print('You must pass exactly three argument!', file=sys.stderr)
        exit(1)
    try:
        noisy_video, threshold_v, neighbors = convert_args_to_parameter(sys.argv[1], sys.argv[2], sys.argv[3])
        start = time.time()
        psnr = dir_w_median_cube(noisy_video, threshold_v, neighbors)
        end = time.time()
        elapsed_time = end - start
        print('PSNR', psnr, 'db')
        print(f'Elapsed time: {elapsed_time:.2f} seconds')
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
