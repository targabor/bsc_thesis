import cv2 as cv
import numpy as np
import os
from . import cpp_calculate


# Image filters
def call_directional_weighted_median_vector(n_image: cv.Mat, threshold: int, height: int, width: int) -> (
        cv.Mat, float):
    filtered_image, psnr = cpp_calculate.directional_weighted_median_vector(n_image, threshold, height, width)
    filtered_image = cv.Mat(np.array(filtered_image).astype(np.uint8))
    return filtered_image, psnr


def call_weighted_median_filter_vector(n_image: cv.Mat, kernel_size: int, weight_type: str = 'uniform') -> (
        cv.Mat, float):
    filtered_image, psnr = cpp_calculate.weighted_median_filter_vector(n_image, kernel_size, weight_type)
    filtered_image = cv.Mat(np.array(filtered_image).astype(np.uint8))
    return filtered_image, psnr


def call_two_pass_median_for_image_vector(n_image: cv.Mat) -> (cv.Mat, float):
    filtered_image, psnr = cpp_calculate.two_pass_median_for_image_vector(n_image)
    filtered_image = cv.Mat(np.array(filtered_image).astype(np.uint8))
    return filtered_image, psnr


def call_basic_median_for_image_vector(n_image: cv.Mat, kernel_size: int) -> (cv.Mat, float):
    filtered_image, psnr = cpp_calculate.basic_median_for_image_vector(n_image, kernel_size)
    filtered_image = cv.Mat(np.array(filtered_image).astype(np.uint8))
    return filtered_image, psnr


# Video noising
def call_add_noise_to_video(video_name: str, noise_percent: float):
    path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    cpp_calculate.add_noise_to_video(path_to_video, video_name, noise_percent)


# Video downscale
def call_downscale_video_res(video_name: str, target_height: int):
    path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    cpp_calculate.downscale_video_res(path_to_video, video_name, target_height)


# Filter for every frames
def call_simple_median_for_video_frame(video_name: str, kernel_size: int, path_to_video='', output_path='') -> float:
    if path_to_video == '':
        path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    else:
        video_name = os.path.basename(path_to_video)
        path_to_video = os.path.dirname(path_to_video) + '/'
        output_path = output_path + '/'
    print(f'{path_to_video=}, {video_name=}, {kernel_size=}, {output_path=}')
    psnr = cpp_calculate.simple_median_for_video_frame(path_to_video, video_name, kernel_size, output_path)
    return psnr


def call_two_pass_median_for_video_frame(video_name: str, path_to_video='', output_path = '') -> float:
    if path_to_video == '':
        path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    else:
        video_name = os.path.basename(path_to_video)
        path_to_video = os.path.dirname(path_to_video) + '/'
        output_path = output_path + '/'
    psnr = cpp_calculate.two_pass_median_for_video_frame(path_to_video, video_name, output_path)
    return psnr


def call_weighted_median_for_video_frame(video_name: str, kernel_size: int, weight_type: str,
                                         path_to_video='') -> float:
    if path_to_video == '':
        path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    else:
        video_name = ''
    psnr = cpp_calculate.weighted_median_for_video_frame(path_to_video, video_name, kernel_size, weight_type)
    return psnr


def call_directional_weighted_median_for_video_frame(video_name: str, threshold: int, path_to_video='') -> float:
    if path_to_video == '':
        path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    else:
        video_name = ''
    psnr = cpp_calculate.directional_weighted_median_for_video_frame(path_to_video, video_name, threshold)
    return psnr


# Filter with respect of other frames
def call_simple_median_cube(video_name: str, kernel: int, neighbors: int, path_to_video='') -> float:
    if path_to_video == '':
        path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    else:
        video_name = ''
    psnr = cpp_calculate.simple_median_cube(path_to_video, video_name, kernel, neighbors)
    return psnr


def call_weighted_median_cube(video_name: str, kernel: int, weight_type: str, neighbors: int,
                              path_to_video='') -> float:
    if path_to_video == '':
        path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    else:
        video_name = ''
    psnr = cpp_calculate.weighted_median_cube(path_to_video, video_name, kernel, weight_type, neighbors)
    return psnr


def call_two_pass_median_cube(video_name: str, neighbors: int, path_to_video='') -> float:
    if path_to_video == '':
        path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    else:
        video_name = ''
    psnr = cpp_calculate.two_pass_median_cube(path_to_video, video_name, neighbors)
    return psnr


def call_dir_w_median_cube(video_name: str, threshold: int, neighbors: int, path_to_video='') -> float:
    if path_to_video == '':
        path_to_video = os.path.dirname(__file__) + '\\..\\..\\videos\\'
    else:
        video_name = ''
    psnr = cpp_calculate.dir_w_median_cube(path_to_video, video_name, threshold, neighbors)
    return psnr
