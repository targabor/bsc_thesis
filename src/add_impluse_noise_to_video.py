import os
import sys

import cpp_calculate


# def add_impulse_noise_to_video(par_video: cv.VideoCapture, video_name: str,  noise_percent: float):
#     """
#     Add impulse noise, for a single grayscale image.
#         :param video_name: name of the original video
#         :param par_video: VideoCapture object of the original video
#         :param noise_percent: represents the percentage of the video we want to distort with noise
#     """
#     fourcc = int(par_video.get(cv.CAP_PROP_FOURCC))
#     fps = int(par_video.get(cv.CAP_PROP_FPS))
#     frame_width = int(par_video.get(cv.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(par_video.get(cv.CAP_PROP_FRAME_HEIGHT))
#     size = (frame_width, frame_height)
#     result = cv.VideoWriter(f'../videos/noisy_{video_name}',
#                             fourcc,
#                             fps,
#                             size,
#                             0)
#     while par_video.isOpened():
#         ret, frame = par_video.read()
#         if not ret:
#             break
#         frame = src.add_impulse_noise_to_img(frame, noise_percent)
#         if cv.waitKey(1) == ord('q'):
#             break
#         result.write(frame)
#
#     par_video.release()
#     result.release()
#     cv.destroyAllWindows()
#
#
# def convert_inputs(video_name: str, percent_str: str) -> (cv.VideoCapture, float):
#     """
#     From the filename, gets the video, and convert the input number to float type.
#         :param video_name: Name of the image in the 'images' folder
#         :param percent_str: String typed floating number, must be between 0 and 1.
#         :return: (Grayscale image (cv2.Mat), floating number between 0 and 1)
#     """
#     __input_video = None
#     try:
#         pattern = re.compile(r"([\w\s]+)\.(mp4)$")
#         if not pattern.match(video_name):
#             raise Exception('The filename must be mp4!')
#         if not os.path.exists(f'../videos/{video_name}'):
#             raise Exception('The file does not exist!')
#         __input_video = cv.VideoCapture(f'../videos/{sys.argv[1]}')
#     except Exception as i_e_f:
#         raise Exception(i_e_f)
#
#     try:
#         __video_percent = float(percent_str)
#
#         if __video_percent > 1 or __video_percent <= 0:
#             raise Exception('The noise_percent must between 1 and 0!')
#
#         return __input_video, __video_percent
#     except Exception as i_e:
#         raise i_e
#
#
# if __name__ == '__main__':
#     input_video = None
#     video_percent = None
#
#     if len(sys.argv) != 3:
#         print('You must pass exactly two argument!', file=sys.stderr)
#         exit(1)
#     try:
#         input_video, video_percent = convert_inputs(sys.argv[1], sys.argv[2])
#     except Exception as e:
#         print(f'There is an error, while processing the arguments!\n{str(e)}', file=sys.stderr)
#         exit(1)
#
#     add_impulse_noise_to_video(input_video, sys.argv[1], video_percent)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
