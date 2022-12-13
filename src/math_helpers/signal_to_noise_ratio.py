import sys
import os
import cv2 as cv
import numpy as np
import re
import mse_for_images


def signal_to_noise_ratio(noisy_image: cv.Mat, filtered_image: cv.Mat) -> float:
    """
    Calculates the signal-to-noise ratio of two images.
    They must have the same .shape .
        :param noisy_image: cv2.Mat object, must be grayscale, the original noisy image
        :param filtered_image: cv2.Mat object, must be grayscale, the filtered image
        :return: The signal-to-noise ratio of the two images.
    """
    variance = (np.var(noisy_image))**2
    mse = mse_for_images.mse_for_images(noisy_image, filtered_image)
    snr = 10 * np.log10(variance/mse)
    return snr


def check_filename(_filename: str) -> bool:
    """
    Check if the given filename is image format and exists.
        :param _filename: filename in string format
        :return: True if the filename is valid and exists, False otherwise
    """
    pattern = re.compile(r"([\w\s]+)\.(jpg|jpeg|png|gif)$")
    assert pattern.match(_filename), f'The file ({_filename}) must be jpg, jpeg or png'
    assert os.path.exists(f'../../images/{_filename}'), f'The given file ({_filename}) does not exists. '
    return True


def check_inputs_for_snr(f_name: str, s_name: str) -> (cv.Mat, cv.Mat):
    """
    Check the two input file, for the snr
        :param f_name: Name of the first image, relative to 'images' folder
        :param s_name: Name of the second image, relative to 'images' folder
        :return: Two cv2.Mat object, which are the two input images.
    """
    if check_filename(f_name) and check_filename(s_name):
        f_image = cv.imread(f'../../images/{f_name}', cv.IMREAD_GRAYSCALE)
        s_image = cv.imread(f'../../images/{s_name}', cv.IMREAD_GRAYSCALE)
        assert len(f_image.shape) == 2, 'The first image is not grayscale'
        assert len(s_image.shape) == 2, 'The second image is not grayscale'
        assert f_image.shape == s_image.shape, 'The images must have the same shape'
        return f_image, s_image
    print('Cannot read the images, from the arguments.\nCheck them and try again', file=sys.stderr)
    exit(1)


# If the program called from commandline
# First is the noisy, second is the filtered
if __name__ == '__main__':
    first_image = None
    second_image = None
    assert len(sys.argv) == 3, 'You must pass exactly 2 parameters'
    try:
        first_image, second_image = check_inputs_for_snr(sys.argv[1], sys.argv[2])
        assert first_image is not None and second_image is not None, 'One of the images is none'
        print(signal_to_noise_ratio(first_image, second_image))
    except Exception as e:
        print(f'There is an error, while processing the arguments!\n{str(e)}', file=sys.stderr)
        exit(1)
    pass
