import re
import os
import time

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from src.add_impulse_noise_to_img import add_impulse_noise_to_img

@profile
def get_w_st(s: int, t: int) -> int:
    valid_range = {-1, 0, 1}
    if s in valid_range and t in valid_range:
        return 2
    return 1

@profile
def calculate_y_plus_s(y: int, s: int, n_image: cv.Mat) -> int:
    height = n_image.shape[0]
    if y + s < 0:
        return 0
    elif y + s >= height:
        return height - 1
    else:
        return y + s

@profile
def calculate_x_plus_t(x: int, t: int, n_image: cv.Mat) -> int:
    width = n_image.shape[1]
    if x + t < 0:
        return 0
    elif x + t >= width:
        return width - 1
    else:
        return x + t

@profile
def own_median(numbers):
    numbers = sorted(numbers)
    length = len(numbers)
    if length % 2 == 0:
        return int(min(np.int64(numbers[length // 2 - 1]) + np.int64(numbers[length // 2]) / 2, 255))
    else:
        return numbers[length // 2]

@profile
def own_argmin(arr):
    min_val = float('inf')
    min_idx = 0
    for i, elem in enumerate(arr):
        if elem < min_val:
            min_val = elem
            min_idx = i
    return min_idx

@profile
def own_std(arr):
    if len(arr) == 0:
        return 0
    mean = sum(arr) / len(arr)
    variance = sum((x - mean) ** 2 for x in arr) / len(arr)
    return variance ** 0.5

@profile
def directional_weighted_median(n_image: cv.Mat, threshold: int) -> cv.Mat:
    coordinates = [
        [(-2, -2), (-1, -1), (1, 1), (2, 2)],  # S_1
        [(0, -2), (0, -1), (0, 1), (0, 2)],  # S_2
        [(2, -2), (1, -1), (-1, 1), (-2, 2)],  # S_3
        [(-2, 0), (-1, 0), (1, 0), (2, 0)],  # S_4
    ]
    o_3 = [
        [(-1, 1), (1, 1)],
        [(0, -1), (0, 1)],
        [(1, -1), (-1, 1)],
        [(-1, 0), (1, 0)]
    ]

    u_ij = np.zeros(n_image.shape)
    height, width = n_image.shape
    for p_num, pixel in enumerate(np.nditer(n_image)):
        y = p_num // width
        x = p_num % width
        d_k = [0] * 4
        std_k = [None] * 4
        for direction in range(4):
            d_sum = 0
            std_dir = [None] * 4
            for num, s in enumerate(coordinates[direction]):
                t = s[1]
                s = s[0]
                w_st = get_w_st(s, t)
                y_plus_s = calculate_y_plus_s(y, s, n_image)
                x_plus_t = calculate_x_plus_t(x, t, n_image)
                w_st_times_abs_y_with_st_minus_y = w_st * abs(n_image[y_plus_s, x_plus_t] - pixel)
                d_sum += w_st_times_abs_y_with_st_minus_y
                std_dir[num] = n_image[y_plus_s, x_plus_t]
            d_k[direction] = d_sum
            std_k[direction] = own_std(std_dir)

        r_ij = int(min(d_k))
        l_ij = own_argmin(std_k)

        alpha_ij = 0
        if r_ij <= threshold:
            alpha_ij = 1
            u_ij[y, x] = pixel
            continue
        values_for_pixel = []
        for direction in range(4):
            for s, t in o_3[direction]:
                w_st = 2 if (s, t) in coordinates[int(l_ij)] else 1
                y_plus_s = calculate_y_plus_s(y, s, n_image)
                x_plus_t = calculate_x_plus_t(x, t, n_image)
                for rep in range(w_st + 1):
                    values_for_pixel.append(n_image[y_plus_s, x_plus_t])
        m_ij = int(own_median(values_for_pixel))
        u_ij[y, x] = alpha_ij * pixel + (1 - alpha_ij) * m_ij
    return cv.Mat(u_ij.astype(np.uint8))


# If the program called from command line
if __name__ == '__main__':
    r_time = []
    noise_percent = 0.1
    for i in range(1, 15):
        t_image = add_impulse_noise_to_img(cv.Mat(np.zeros((2 ** i, 2 ** i)).astype(np.uint8)), noise_percent)
        start = time.time()
        filtered = directional_weighted_median(t_image, 200)
        end = time.time()
        r_time.append(end - start)
    plt.plot(range(1, 15), r_time)
    plt.show()
