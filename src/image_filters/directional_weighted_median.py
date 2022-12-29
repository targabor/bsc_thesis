import sys
import re
import os

import cv2 as cv
import numpy as np

from src.math_helpers.signal_to_noise_ratio import signal_to_noise_ratio


def get_w_st(s: int, t: int) -> int:
    """
    Returns the reciprocal of distance ratio.
    2, if -1 <= s,t <= 1
    1, otherwise
        :param s: first offset, whole number
        :param t: second offset, whole number
        :return: whole number 1 or 2
    """
    if -1 <= s <= 1 and -1 <= t <= 1:
        return 2
    return 1


def calculate_y_plus_s(y: int, s: int, n_image: cv.Mat) -> int:
    height = n_image.shape[0]
    return int(np.clip(y + s, 0, height - 1))


def calculate_x_plus_t(x: int, t: int, n_image: cv.Mat) -> int:
    width = n_image.shape[1]
    return int(np.clip(x + t, 0, width - 1))


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
    r_ij = np.zeros(n_image.shape)
    l_ij = np.zeros(n_image.shape)
    m_ij = np.zeros(n_image.shape)
    u_ij = np.zeros(n_image.shape)

    for y in range(n_image.shape[0]):
        for x in range(n_image.shape[1]):
            d_k = []
            std_k = []
            for direction in range(4):
                d_sum = 0
                std_dir = []
                for s, t in coordinates[direction]:
                    w_st = get_w_st(s, t)
                    y_plus_s = calculate_y_plus_s(y, s, n_image)
                    x_plus_t = calculate_x_plus_t(x, t, n_image)
                    w_st_times_abs_y_with_st_minus_y = w_st * np.abs(n_image[y_plus_s, x_plus_t] - n_image[y, x])
                    d_sum += w_st_times_abs_y_with_st_minus_y
                    std_dir.append(n_image[y_plus_s, x_plus_t])
                d_k.append(d_sum)
                std_k.append(np.std(std_dir))
            r_ij[y, x] = int(np.min(d_k))
            l_ij[y, x] = np.argmin(std_k)

    for y in range(n_image.shape[0]):
        for x in range(n_image.shape[1]):
            values_for_pixel = []
            if r_ij[y, x] <= threshold:
                continue
            for direction in range(4):
                for s, t in o_3[direction]:
                    w_st = 2 if (s, t) in coordinates[int(l_ij[y, x])] else 1
                    y_plus_s = calculate_y_plus_s(y, s, n_image)
                    x_plus_t = calculate_x_plus_t(x, t, n_image)
                    for rep in range(w_st + 1):
                        values_for_pixel.append(n_image[y_plus_s, x_plus_t])
            m_ij[y, x] = int(np.median(values_for_pixel))
            alpha_ij = 0 if r_ij[y, x] > threshold else 1
            u_ij[y, x] = alpha_ij * n_image[y, x] + (1 - alpha_ij) * m_ij[y, x]

    u_ij[r_ij > threshold] = n_image[r_ij>threshold]
    return cv.Mat(u_ij.astype(np.uint8))


def convert_inputs_for_weighted(image_name: str) -> cv.Mat:
    """
    From the filename, gets the image and checks, and converts it to grayscale
        :param image_name: Name of the image in the 'images' folder
        :return: Grayscale image (cv2.Mat)
    """
    __input_image = None
    try:
        pattern = re.compile(r"([\w\s]+)\.(jpg|jpeg|png)$")
        if not pattern.match(image_name):
            raise Exception('The filename must be jpg, jpeg, png!')
        if not os.path.exists(f'../../images/{image_name}'):
            raise Exception('The file does not exist!')
        __input_image = cv.imread(f'../../images/{image_name}')
        # To make sure, the input will be grayscale
        __input_image = cv.cvtColor(__input_image, cv.COLOR_BGR2GRAY)
    except Exception as i_e_f:
        raise Exception(i_e_f)

    return __input_image


# If the program called from command line
if __name__ == '__main__':
    input_image = None
    if len(sys.argv) != 3:
        print('You must pass exactly two argument!', file=sys.stderr)
        exit(1)
    assert sys.argv[2].isdigit(), 'The second parameter should be an integer'

    try:
        image = convert_inputs_for_weighted(sys.argv[1])
        cv.imshow('input_image', image)
        if image is not None:
            filtered_image = directional_weighted_median(image, int(sys.argv[2]))
            cv.imshow('filtered_image', filtered_image)
            print('signal_to_noise_filtered', signal_to_noise_ratio(image, filtered_image))
            cv.waitKey()
            cv.destroyAllWindows()
            exit(0)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)
