import os
import sys
import cv2 as cv
import re
from src.dlls import cpp_caller


def convert_inputs(video_name: str, height_string: str) -> (cv.VideoCapture, int):
    """
    From the filename, gets the video, and convert the input number to float type.
        :param video_name: Name of the image in the 'images' folder
        :param percent_str: String typed floating number, must be between 0 and 1.
        :return: (name of the video (cv2.Mat), floating number between 0 and 1)
    """
    __input_video = None
    try:
        pattern = re.compile(r"([\w\s]+)\.(mp4)$")
        if not pattern.match(video_name):
            raise Exception('The filename must be mp4!')
        if not os.path.exists(f'../videos/{video_name}'):
            raise Exception('The file does not exist!')
    except Exception as i_e_f:
        raise Exception(i_e_f)

    try:
        assert height_string.isdigit(), 'The second parameter must be digit'
        assert float(height_string) % 1 == 0, 'The second parameter must be whole number'
        height = int(height_string)
        assert height > 0, 'Second parameter must be greater than 0'

        return video_name, height
    except Exception as i_e:
        raise i_e


if __name__ == '__main__':
    input_video = None
    target_height = None
    if len(sys.argv) != 3:
        print('You must pass exactly two argument!', file=sys.stderr)
        exit(1)
    try:
        input_video, target_height = convert_inputs(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f'There is an error, while processing the arguments!\n{str(e)}', file=sys.stderr)
        exit(1)

    cpp_caller.call_downscale_video_res(input_video, target_height)
    cv.waitKey(0)
    cv.destroyAllWindows()
