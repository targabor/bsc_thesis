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
