import time

import cv2 as cv
import numpy as np

import sys
import os
import re

from src.dlls import cpp_caller


def directional_weighted_median_for_video_frame(video_name: str, threshold: int) -> float:
    return cpp_caller.call_directional_weighted_median_for_video_frame(video_name, threshold)


def convert_args_to_parameter(video_name: str, threshold_str: str) -> (str, int):
    """
    Converts the command line arguments to VideoCapture object and int for kernel size
        :param threshold_str: threshold value in string
        :param video_name: video filename, from videos folder, must be grayscale
        :return: VideoCapture object from 'videos/video_name' and integer
    """
    __input_video = None
    threshold = 0
    try:
        pattern = re.compile(r"([\w\s]+)\.(mp4)$")
        if not pattern.match(video_name):
            raise Exception('The filename must be mp4')
        if not os.path.exists(f'../../videos/{video_name}'):
            raise Exception('The file does not exist!')
        assert threshold_str.isdigit(), 'the second parameter must be a digit!'
        threshold = int(threshold_str)
        assert threshold >= 0, 'the threshold must be a positive number'
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return video_name, threshold


if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 3:
        print('You must pass exactly two argument!', file=sys.stderr)
        exit(1)
    try:
        noisy_video, threshold = convert_args_to_parameter(sys.argv[1], sys.argv[2])
        start = time.time()
        psnr = directional_weighted_median_for_video_frame(noisy_video, threshold)
        end = time.time()
        elapsed_time = end - start
        print('PSNR', psnr, 'db')
        print(f'Elapsed time: {elapsed_time:.2f} seconds')
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
