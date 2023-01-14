import time

import cv2 as cv
import numpy as np

import sys
import os
import re

from src.dlls import cpp_caller


def two_pass_median_for_video_frame(noisy_videoname: str) -> float:
    return cpp_caller.call_two_pass_median_for_video_frame(noisy_videoname)


def convert_args_to_parameter(video_name: str) -> str:
    """
    Converts the command line arguments to VideoCapture object and int for kernel size
        :param video_name: video filename, from videos folder, must be grayscale
        :return: video_name if it's valid
    """
    try:
        pattern = re.compile(r"([\w\s]+)\.(mp4)$")
        if not pattern.match(video_name):
            raise Exception('The filename must be mp4')
        if not os.path.exists(f'../../videos/{video_name}'):
            raise Exception('The file does not exist!')
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return video_name


# if the program called from the command line
if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 2:
        print('You must pass exactly one argument!', file=sys.stderr)
        exit(1)
    try:
        noisy_video = convert_args_to_parameter(sys.argv[1])
        start = time.time()
        psnr = two_pass_median_for_video_frame(noisy_video)
        end = time.time()
        elapsed_time = end - start
        print('PSNR', psnr, 'db')
        print(f'Elapsed time: {elapsed_time:.2f} seconds')
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
