import time

import cv2 as cv
import numpy as np

import sys
import os
import re

from src.dlls import cpp_caller


def convert_args_to_parameter(video_name: str, kernel_str: str) -> (str, int):
    """
    Converts the command line arguments to VideoCapture object and int for kernel size
        :param video_name: video filename, from videos folder, must be grayscale
        :param kernel_str: kernel size, must be odd number, and greater than 1 (3,5,7, etc.)
        :return: VideoCapture object from 'videos/video_name' and interger
    """
    kernel_size = 0
    try:
        pattern = re.compile(r"([\w\s]+)\.(mp4)$")
        if not pattern.match(video_name):
            raise Exception('The filename must be mp4')
        if not os.path.exists(f'../../videos/{video_name}'):
            raise Exception('The file does not exist!')
        assert kernel_str.isdigit(), 'the second parameter must be a digit!'
        kernel_size = int(kernel_str)
        assert kernel_size % 2 == 1, 'the second parameter must be odd'
        assert kernel_size > 1, 'the second parameter must be greater than 1'
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return video_name, kernel_size


# if the program called from the command line
if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 3:
        print('You must pass exactly two argument!', file=sys.stderr)
        exit(1)
    try:
        noisy_video, kernel = convert_args_to_parameter(sys.argv[1], sys.argv[2])
        start = time.time()
        cpp_caller.call_simple_median_for_video_frame(noisy_video, kernel)
        end = time.time()
        elapsed_time = end - start
        print(f'Elapsed time: {elapsed_time:.2f} seconds')
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)