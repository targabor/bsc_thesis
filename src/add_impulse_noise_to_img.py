from random import randint

import cv2 as cv
import sys
import re
import os


def add_impulse_noise_to_img(par_img: cv.Mat, noise_percent: float) -> cv.Mat:
    """
    Add impulse noise, for a single grayscale image.
        :param par_img: OpenCV grayscale image
        :param noise_percent: represents the percentage of the image we want to distort with noise
        :return: Image, with added impulse noise
    """
    ret_img = par_img.copy()
    mask_size = round(ret_img.size * noise_percent)
    for i in range(mask_size):
        x = randint(0, ret_img.shape[1] - 1)
        y = randint(0, ret_img.shape[0] - 1)
        ret_img[y, x] = randint(0, 255)

    return ret_img


def __convert_inputs(image_name: str, percent_str: str) -> (cv.Mat, float):
    try:
        pattern = re.compile(r"([\w\s]+)\.(jpg|jpeg|png|gif)$")
        if not pattern.match(image_name):
            raise Exception('The filename must be jpg, jpeg, png or gif!')
        if not os.path.exists(f'../images/{image_name}'):
            print(os.path.join(__file__, '..', '..','images', image_name))
            raise Exception('The file does not exist!')
        __input_image = cv.imread(f'../images/{image_name}')
    except Exception as i_e_f:
        raise Exception(i_e_f)

    try:
        __img_percent = float(percent_str)

        if __img_percent > 1 or __img_percent <= 0:
            raise Exception('The noise_percent must between 1 and 0!')

        return __input_image, __img_percent
    except Exception as i_e:
        raise i_e


if __name__ == '__main__':
    input_image = None
    img_percent = None

    if len(sys.argv) != 3:
        print('You must pass exactly two argument!', file=sys.stderr)
        exit(1)
    try:
        input_image, img_percent = __convert_inputs(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f'There is an error, while processing the arguments!\n{str(e)}', file=sys.stderr)
        exit(1)

    cv.imshow('input', input_image)
    noised_image = add_impulse_noise_to_img(input_image, img_percent)
    cv.imshow('output', noised_image)
    cv.imwrite('beboon_noise.jpg', noised_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
