import cv2 as cv
import numpy as np
import os
import ctypes
from . import cpp_calculate


def call_directional_weighted_median(n_image: cv.Mat, threshold: int, height: int, width: int) -> cv.Mat:
    filtered_image = cpp_calculate.directional_weighted_median(n_image, threshold, height, width)
    filtered_image = cv.Mat(np.array(filtered_image).astype(np.uint8))
    return filtered_image


def call_add_noise_to_video(video_name: str, noise_percent: float):
    path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    cpp_calculate.add_noise_to_video(path_to_video, video_name, noise_percent)


def call_simple_median_for_video(video_name: str, kernel_size: int):
    path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    cpp_calculate.simple_median_for_video(path_to_video, video_name, kernel_size)


def call_two_pass_median_for_image_vector(n_image: cv.Mat) -> cv.Mat:
    filtered_image = cpp_calculate.two_pass_median_for_image_vector(n_image)
    filtered_image = cv.Mat(np.array(filtered_image).astype(np.uint8))
    return filtered_image


def call_weighted_median_filter_vector(n_image: cv.Mat, kernel_size: int, weight_type: str = 'uniform') -> cv.Mat:
    filtered_image = cpp_calculate.weighted_median_filter_vector(n_image, kernel_size, weight_type)
    filtered_image = cv.Mat(np.array(filtered_image).astype(np.uint8))
    return filtered_image


