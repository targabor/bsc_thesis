import os
import sys
import cv2 as cv
import re
from src.dlls import cpp_caller


def convert_inputs(video_name: str, percent_str: str) -> (cv.VideoCapture, float):
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
        __video_percent = float(percent_str)

        if __video_percent > 1 or __video_percent <= 0:
            raise Exception('The noise_percent must between 1 and 0!')

        return video_name, __video_percent
    except Exception as i_e:
        raise i_e


if __name__ == '__main__':
    input_video = None
    video_percent = None
    if len(sys.argv) != 3:
        print('You must pass exactly two argument!', file=sys.stderr)
        exit(1)
    try:
        input_video, video_percent = convert_inputs(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f'There is an error, while processing the arguments!\n{str(e)}', file=sys.stderr)
        exit(1)

    cpp_caller.call_add_noise_to_video(input_video, video_percent)
    cv.waitKey(0)
    cv.destroyAllWindows()
