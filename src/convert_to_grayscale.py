import cv2 as cv
import os
import re
import sys


def check_filename(_filename: str) -> bool:
    """
    Check if the given filename is image format and exists.
        :param _filename: filename in string format
        :return: True if the filename is valid and exists, False otherwise
    """
    pattern = re.compile(r"([\w\s]+)\.(jpg|jpeg|png|gif)$")
    assert pattern.match(_filename), f'The file ({_filename}) must be jpg, jpeg or png'
    assert os.path.exists(f'../images/{_filename}'), f'The given file ({_filename}) does not exists. '
    return True


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'You must pass exactly one parameter'
    if check_filename(sys.argv[1]):
        original = cv.imread(f'../images/{sys.argv[1]}')
        gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
        cv.imwrite(f'../images/gray_{sys.argv[1]}', gray)
        print('done')
