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
    pattern = re.compile(r"([\w\s]+)\.(mp4)$")
    assert pattern.match(_filename), f'The file ({_filename}) must be jpg, jpeg or png'
    assert os.path.exists(f'../videos/{_filename}'), f'The given file ({_filename}) does not exists. '
    return True


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'You must pass exactly one parameter'
    if check_filename(sys.argv[1]):
        original = cv.VideoCapture(f'../videos/{sys.argv[1]}')
        fourcc = int(original.get(cv.CAP_PROP_FOURCC))
        fps = int(original.get(cv.CAP_PROP_FPS))
        frame_width = int(original.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(original.get(cv.CAP_PROP_FRAME_HEIGHT))
        size = (frame_width, frame_height)
        result = cv.VideoWriter(f'../videos/gray_{sys.argv[1]}',
                                fourcc,
                                fps,
                                size,
                                0)
        while original.isOpened():
            ret, frame = original.read()
            if not ret:
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
            result.write(frame)

        original.release()
        result.release()
        cv.destroyAllWindows()
