from src.dlls import cpp_caller
import os
import cv2 as cv


def calculate_ssim_for_videos(original_path: str, filtered_path: str) -> float:
    print(f'{original_path=} \t {filtered_path=}')
    original_video = cv.VideoCapture(original_path)
    if not original_video.isOpened():
        print('Cannot open video')
        exit()
    original_video.set(cv.CAP_PROP_CONVERT_RGB, .0)

    filtered_video = cv.VideoCapture(filtered_path)
    if not filtered_video.isOpened():
        print('Cannot open video')
        exit()
    filtered_video.set(cv.CAP_PROP_CONVERT_RGB, 0.)

    ssim_sum = 0.0
    frame_count = 0
    while True:
        orig_ret, orig_frame = original_video.read()
        filt_ret, filt_frame = filtered_video.read()
        if not orig_ret and not filt_ret:
            break

        ssim_score, _ = cv.quality.QualitySSIM_compute(orig_frame, filt_frame)
        ssim_sum += ssim_score[0]
        frame_count += 1

    return ssim_sum / frame_count


test_input_video_path = 'measures/gray_luna_bimbi.mp4'

noise_output_path = 'measures/noised/'

files = os.listdir(noise_output_path)
file_paths = []  # Noisy files

for file in files:
    file_path = os.path.join(noise_output_path, file)
    file_paths.append(file_path)

with open("measure_result.txt", "w", buffering=1024) as file:
    # Simple median every frame
    for file_path in file_paths:
        output_folder_path = 'measures/simple_median_every_frame'
        for kernel in range(3, 11, 2):
            psnr_for_noisy_and_filtered = cpp_caller.call_simple_median_for_video_frame('', kernel, file_path,
                                                                                        output_folder_path)  # higher is better
            ssim = calculate_ssim_for_videos(test_input_video_path,
                                             output_folder_path + '/' + f'simple_median_{kernel}_{os.path.basename(file_path)}')  # higher is better
            file.write(f'Simple median filter with {kernel} kernel size ended for {file_path}\n'
                       f'Stats:\n'
                       f'PSNR: {psnr_for_noisy_and_filtered} db\n'
                       f'SSIM: {ssim}\n\n')
            

        # Two pass every frame
        output_folder_path = 'measures/two_pass_every_frame'
        psnr_for_noisy_and_filtered = cpp_caller.call_two_pass_median_for_video_frame('', file_path,
                                                                                      output_folder_path)  # higher is better
        ssim = calculate_ssim_for_videos(test_input_video_path,
                                         output_folder_path + '/' + f'two_pass_median_{os.path.basename(file_path)}')  # higher is better
        file.write(f'Two Pass median filter ended for {file_path}\n'
                   f'Stats:\n'
                   f'PSNR: {psnr_for_noisy_and_filtered} db\n'
                   f'SSIM: {ssim}\n\n')

        # Weighted median every frame
        output_folder_path = 'measures/weighted_every_frame'
        for kernel in range(3, 11, 2):
            for weight_type in ('uniform', 'distance'):
                # def call_weighted_median_for_video_frame(video_name: str, kernel_size: int, weight_type: str,
                #                                          path_to_video='', output_path='') -> float:
                psnr_for_noisy_and_filtered = cpp_caller.call_weighted_median_for_video_frame('', kernel, weight_type,
                                                                                              file_path,
                                                                                              output_folder_path)  # higher is better
                ssim = calculate_ssim_for_videos(test_input_video_path,
                                                 output_folder_path + '/' + f'weighted_median_{weight_type}_{kernel}_{os.path.basename(file_path)}')  # higher is better
                file.write(
                    f'Weighted median filter with {kernel} kernel size and {weight_type} weight type ended for {file_path}\n'
                    f'Stats:\n'
                    f'PSNR: {psnr_for_noisy_and_filtered} db\n'
                    f'SSIM: {ssim}\n\n')
                

        # Directional weighted every frame
        output_folder_path = 'measures/directional_weighted_every_frame'
        threshold = 510
        while threshold > 100:
            psnr_for_noisy_and_filtered = cpp_caller.call_directional_weighted_median_for_video_frame('',
                                                                                                      threshold,
                                                                                                      file_path,
                                                                                                      output_folder_path)  # higher is better
            ssim = calculate_ssim_for_videos(test_input_video_path,
                                             output_folder_path + '/' + f'dir_w_median_{threshold}_{os.path.basename(file_path)}')  # higher is better
            file.write(
                f'Directional weighted median filter with {threshold} threshodld ended for {file_path}\n'
                f'Stats:\n'
                f'PSNR: {psnr_for_noisy_and_filtered} db\n'
                f'SSIM: {ssim}\n\n')
            
            threshold = int(threshold * 0.8)
        # Simple median cube
        output_folder_path = 'measures/simple_cube'
        for frame_n in range(1, 6):
            for kernel in range(3, 11, 2):
                psnr_for_noisy_and_filtered = cpp_caller.call_simple_median_cube('', kernel, frame_n,
                                                                                 file_path,
                                                                                 output_folder_path)  # higher is better
                ssim = calculate_ssim_for_videos(test_input_video_path,
                                                 output_folder_path + '/' + f'simple_cube_{frame_n}_{kernel}_{os.path.basename(file_path)}')  # higher is better
                file.write(
                    f'Simple median cube with {frame_n} neighbors and {kernel} kernel size ended for {file_path}\n'
                    f'Stats:\n'
                    f'PSNR: {psnr_for_noisy_and_filtered} db\n'
                    f'SSIM: {ssim}\n\n')
                

        # Two pass median cube
        output_folder_path = 'measures/two_pass_cube'
        for frame_n in range(1, 6):
            psnr_for_noisy_and_filtered = cpp_caller.call_two_pass_median_cube('', frame_n, file_path,
                                                                               output_folder_path)  # higher is better
            ssim = calculate_ssim_for_videos(test_input_video_path,
                                             output_folder_path + '/' + f'two_pass_median_{frame_n}_{os.path.basename(file_path)}')  # higher is better
            file.write(
                f'Two Pass cube with {frame_n} neighbors ended for {file_path}\n'
                f'Stats:\n'
                f'PSNR: {psnr_for_noisy_and_filtered} db\n'
                f'SSIM: {ssim}\n\n')
            

        # Weighted median cube
        output_folder_path = 'measures/weighted_cube'
        for frame_n in range(1, 6):
            for kernel in range(3, 11, 2):
                for weight_type in ('uniform', 'distance'):
                    psnr_for_noisy_and_filtered = cpp_caller.call_weighted_median_cube('', kernel, weight_type, frame_n,
                                                                                       file_path,
                                                                                       output_folder_path)  # higher is better
                    ssim = calculate_ssim_for_videos(test_input_video_path,
                                                     output_folder_path + '/' + f'weighted_median_{frame_n}_{kernel}_{weight_type}_{os.path.basename(file_path)}')  # higher is better
                    file.write(
                        f'Weighted median cube with {frame_n} neighbors and {kernel} kernel size and {weight_type} weight type ended for {file_path}\n'
                        f'Stats:\n'
                        f'PSNR: {psnr_for_noisy_and_filtered} db\n'
                        f'SSIM: {ssim}\n\n')
                    

        # Directional weighted median cube
        output_folder_path = 'measures/directional_cube'
        for frame_n in range(1, 6):
            threshold = 510
            while threshold > 100:
                psnr_for_noisy_and_filtered = cpp_caller.call_dir_w_median_cube('', threshold, frame_n, file_path,
                                                                                output_folder_path)  # higher is better
                ssim = calculate_ssim_for_videos(test_input_video_path,
                                                 output_folder_path + '/' + f'dir_w_cube_{frame_n}_{threshold}_{os.path.basename(file_path)}')  # higher is better
                file.write(
                    f'Directional weighted median cube with {frame_n} neighbors and {threshold} threshodld ended for {file_path}\n'
                    f'Stats:\n'
                    f'PSNR: {psnr_for_noisy_and_filtered} db\n'
                    f'SSIM: {ssim}\n\n')
                
                threshold = threshold * 0.8
