#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include <filesystem>
#include <cstdlib>

namespace py = pybind11;

cv::Mat convert_vector_to_mat(std::vector<std::vector<int>> n_image){
  int rows = n_image.size();
  int cols = n_image[0].size();
  cv::Mat myMat(rows, cols, CV_8UC1);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        myMat.at<uchar>(i, j) = (uchar)n_image[i][j];
    }
  }
  return myMat;
}

std::vector<std::vector<int>> convert_mat_to_vector(cv::Mat myMat){
  std::vector<std::vector<int>> data(myMat.rows, std::vector<int>(myMat.cols));
  for (int i = 0; i < myMat.rows; i++) {  
    for (int j = 0; j < myMat.cols; j++) {
      data[i][j] = myMat.at<uchar>(i, j);
    }
  }
  return data;
}

cv::Mat weighted_median_filter_mat(cv::Mat image, int kernel_size, std::string weight_type = "uniform") {
    int padding = kernel_size / 2;
    cv::Mat padded_image;
    cv::copyMakeBorder(image, padded_image, padding, padding, padding, padding, cv::BORDER_REPLICATE);
    cv::Mat output_image = cv::Mat::zeros(image.rows, image.cols, image.type());
    cv::Mat weights;
    if (weight_type == "uniform") {
        weights = cv::Mat::ones(kernel_size, kernel_size, CV_32FC1);
    } else if (weight_type == "distance") {
        int center = (kernel_size - 1) / 2;
        weights = cv::Mat::zeros(kernel_size, kernel_size, CV_32FC1);
        for (int x = 0; x < kernel_size; x++) {
            for (int y = 0; y < kernel_size; y++) {
                float distance = sqrt(pow(x - center, 2) + pow(y - center, 2));
                weights.at<float>(y, x) = exp(-distance / kernel_size);
            }
        }
    } else {
        throw std::invalid_argument("Invalid weight type. Use 'uniform' or 'distance'.");
    }
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Rect roi(x, y, kernel_size, kernel_size);
            cv::Mat neighborhood = padded_image(roi);
            std::vector<int> values;
            for (int i = 0; i < neighborhood.rows; i++) {
                for (int j = 0; j < neighborhood.cols; j++) {
                    float weight = weights.at<float>(i, j);
                    for (int k = 0; k < weight; k++) {
                        values.push_back(neighborhood.at<uchar>(i, j));
                    }
                }
            }
            std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
            output_image.at<uchar>(y, x) = values[values.size() / 2];
        }
    }
    return output_image;
}

std::vector<std::vector<int>> weighted_median_filter_vector(std::vector<std::vector<int>> n_image, int kernel_size, std::string weight_type = "uniform"){
  cv::Mat myMat = convert_vector_to_mat(n_image);
  myMat = weighted_median_filter_mat(myMat, kernel_size, weight_type);
  return convert_mat_to_vector(myMat);
}

cv::Mat generate_noise_map(cv::Mat _apx_of_noise) {
  cv::Scalar mean, std_dev;
  cv::meanStdDev(_apx_of_noise, mean, std_dev);
  double sum = cv::sum(_apx_of_noise)[0];
  double num_pixels = _apx_of_noise.rows * _apx_of_noise.cols;
  double avg_grey_level = sum / num_pixels;
  double std_deviation = std_dev[0];
  cv::Mat noise_map;
  cv::compare(_apx_of_noise, std_deviation + avg_grey_level, noise_map, cv::CMP_GE);
  return noise_map;
}

cv::Mat two_pass_median_for_image_mat(const cv::Mat image_to_filter){
  cv::Mat apx_of_signal_only;
  cv::medianBlur(image_to_filter, apx_of_signal_only, 3);
  cv::Mat apx_of_noise;
  cv::absdiff(image_to_filter, apx_of_signal_only, apx_of_noise);
  cv::Mat noise_map = generate_noise_map(apx_of_noise);
  cv::Mat _filtered_image = image_to_filter.clone();
  apx_of_signal_only.copyTo(_filtered_image, noise_map);
  cv::waitKey();
  return _filtered_image;
}

std::vector<std::vector<int>> two_pass_median_for_image_vector(std::vector<std::vector<int>> n_image){
  cv::Mat myMat = convert_vector_to_mat(n_image);
  myMat = two_pass_median_for_image_mat(myMat);
  return convert_mat_to_vector(myMat);
}

void simple_median_for_video(const std::string &video_path, const std::string &video_name, int kernel_size){
  cv::VideoCapture capture(video_path + video_name);
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC); 
  cv::VideoWriter writer((video_path + "simple_median_" + video_name), fourcc, fps, cv::Size(width, height),0);

  if (!capture.isOpened()) {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
  }
  cv::Mat frame;
  while (capture.read(frame)) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::medianBlur(frame, frame, kernel_size);
    writer.write(frame);
    if (cv::waitKey(33) >= 0) {
      break;
    }
  }
  capture.release();
  writer.release();
  std::cout << "done" << std::endl;
}

const cv::Mat& add_noise_to_frame(cv::Mat &frame, float noise_percent){
  int pixels_to_be_noised = std::round(frame.total() * noise_percent);
  for(int count = 0; count < pixels_to_be_noised; count++){
    int randomWidth = rand() % (frame.cols - 1); 
    int randomHeight = rand() % (frame.rows - 1 );
    frame.at<uchar>(randomHeight, randomWidth) = rand() % 255; 
  }
  return frame;
}

void add_noise_to_video(const std::string &video_path, const std::string &video_name, float noise_percent){
  cv::VideoCapture capture(video_path + video_name);
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC); 
  cv::VideoWriter writer((video_path + "noisy_" + video_name), fourcc, fps, cv::Size(width, height),0);

  if (!capture.isOpened()) {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
  }
  cv::Mat frame;
  while (capture.read(frame)) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frame = add_noise_to_frame(frame, noise_percent);
    writer.write(frame);
    if (cv::waitKey(33) >= 0) {
      break;
    }
  }
  capture.release();
  writer.release();
  std::cout << "done" << std::endl;
}

float own_median(std::vector<int> numbers) {
    std::sort(numbers.begin(), numbers.end());

    int size = numbers.size();
    double median = 0;
    if (size % 2 == 0) {
        median = (numbers[size / 2 - 1] + numbers[size / 2]) / 2.0;
    } else {
        median = numbers[size / 2];
    }

  return median;
}

int get_w_st(int s, int t) {
  switch (s) {
    case -1:
    case 0:
    case 1:
      if (t == -1 || t == 0 || t == 1) {
        return 2;
      }
      break;
  }
  return 1;
}

int calculate_y_plus_s(int y, int s, int height) {
  if (y + s < 0) {
    return 0;
  } else if (y + s >= height) {
    return height - 1;
  } else {
    return y + s;
  }
}

int calculate_x_plus_t(int x, int t, int width) {
  if (x + t < 0) {
    return 0;
  } else if (x + t >= width) {
    return width - 1;
  } else {
    return x + t;
  }
}

int own_argmin(std::vector<int>& arr) {
  auto min_it = std::min_element(arr.begin(), arr.end());
  return std::distance(arr.begin(), min_it);
}


double own_mean(std::vector<int>& arr) {
  if (arr.empty()) {
    return 0;
  }
  double sum = 0;
  for (int x : arr) {
    sum += x;
  }
  return sum / arr.size();
}

double own_std(std::vector<int>& arr) {
  if (arr.empty()) {
    return 0;
  }
  double mean = own_mean(arr);
  double variance = 0;
  for (int x : arr) {
    variance += (x - mean) * (x - mean);
  }
  variance /= arr.size();
  return std::sqrt(variance);
}

int is_s_t_in_coordinates(int s, int t, int l_ij, std::vector<std::vector<std::pair<int, int>>>& coordinates){
  for (const auto& direction : coordinates[l_ij]) {
    if(direction.first == s && direction.second == t){
      return 2;
    }
  }
  return 1;
}

std::vector<std::vector<int>>  directional_weighted_median(std::vector<std::vector<int>> n_image, int threshold, int height, int width) {
  std::vector<std::vector<std::pair<int, int>>> coordinates = {
    {{-2, -2}, {-1, -1}, {1, 1}, {2, 2}},  // S_1
    {{0, -2}, {0, -1}, {0, 1}, {0, 2}},  // S_2
    {{2, -2}, {1, -1}, {-1, 1}, {-2, 2}},  // S_3
    {{-2, 0}, {-1, 0}, {1, 0}, {2, 0}},  // S_4
  };
  std::vector<std::vector<std::pair<int, int>>> o_3 = {
    {{-1, 1}, {1, 1}},
    {{0, -1}, {0, 1}},
    {{1, -1}, {-1, 1}},
    {{-1, 0}, {1, 0}}
  };

  std::vector<std::vector<int>> u_ij(height, std::vector<int>(width));
  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      std::vector<int> d_k(4, 0);
      std::vector<int> std_k(4, 0);
      for(int direction = 0; direction < 4; direction++){
        int d_sum = 0;
        std::vector<int> dir_std(4, 0);
        int counter = 0;
        for (const auto& point : coordinates[direction]) {
          int s = point.first;
          int t = point.second;
          int w_st = get_w_st(s,t);
          int y_plus_s = calculate_y_plus_s(y,s,height);
          int x_plus_t = calculate_x_plus_t(x,t,width);
          int w_st_times_abs_y_with_st_minus_y = w_st * std::abs(n_image[y_plus_s][x_plus_t] - n_image[y][x]);
          d_sum += w_st_times_abs_y_with_st_minus_y;
          dir_std[counter] = n_image[y_plus_s][x_plus_t];
          counter++;
        }
        d_k[direction] = d_sum;
        std_k[direction] = own_std(dir_std);
      }

      int r_ij = *std::min_element(d_k.begin(), d_k.end());
      int l_ij = own_argmin(std_k);

      int alpha_ij = 0;
      if(r_ij <= threshold){
        u_ij[y][x] = n_image[y][x];
        continue;
      }

      std::vector<int> values_for_pixel;
      for(int direction = 0; direction < 4; direction++){
        for (const auto& point : o_3[direction]) {
          int s = point.first;
          int t = point.second;
          int w_st = is_s_t_in_coordinates(s,t, l_ij, coordinates);
          int y_plus_s = calculate_y_plus_s(y,s,height);
          int x_plus_t = calculate_x_plus_t(x,t,width);
          for(int rep = 0; rep < w_st; rep++){
            values_for_pixel.push_back(n_image[y_plus_s][x_plus_t]);
          }
        }
      }
      u_ij[y][x] = own_median(values_for_pixel);
    }
  }
  return u_ij;
}

PYBIND11_MODULE(cpp_calculate, module_handle) {
    module_handle.doc() = "I'm a docstring hehe";
    module_handle.def("directional_weighted_median", &directional_weighted_median);
    module_handle.def("add_noise_to_video", &add_noise_to_video);
    module_handle.def("simple_median_for_video", &simple_median_for_video);
    module_handle.def("two_pass_median_for_image_vector", &two_pass_median_for_image_vector);
    module_handle.def("weighted_median_filter_vector", &weighted_median_filter_vector);
}