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
#include <chrono>
#include <filesystem>
#include <cstdlib>

namespace py = pybind11;
// Helpres----------------------------------------------------------------------------------------------

void downscale_video_res(std::string &video_path, std::string &video_name, int height)
{
  cv::VideoCapture capture(video_path + video_name);
  int o_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int o_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  int target_width = std::round(height * ((float)o_width / o_height));
  cv::Size frame_size(target_width, height);
  cv::VideoWriter writer((video_path + "decreased_to_" + std::to_string(height) + "_" + video_name), fourcc, fps, frame_size, 0);

  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
  }
  cv::Mat frame;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::resize(frame, frame, frame_size, 0, 0, cv::INTER_LINEAR);
    writer.write(frame);
    if (cv::waitKey(33) >= 0)
    {
      break;
    }
  }
  capture.release();
  writer.release();
}

double PSNR(cv::Mat &original, cv::Mat &compressed)
{
  cv::Scalar mse = cv::mean((original - compressed).mul(original - compressed));
  if (mse[0] == 0)
  {
    return 100;
  }
  double max_pixel = 255.0;
  double psnr = 20 * log10(max_pixel / sqrt(mse[0]));
  return psnr;
}

cv::Mat convert_vector_to_mat(std::vector<std::vector<int>> n_image)
{
  int rows = n_image.size();
  int cols = n_image[0].size();
  cv::Mat myMat(rows, cols, CV_8UC1);
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      myMat.at<uchar>(i, j) = (uchar)n_image[i][j];
    }
  }
  return myMat;
}

std::vector<std::vector<int>> convert_mat_to_vector(cv::Mat myMat)
{
  std::vector<std::vector<int>> data(myMat.rows, std::vector<int>(myMat.cols));
  for (int i = 0; i < myMat.rows; i++)
  {
    for (int j = 0; j < myMat.cols; j++)
    {
      data[i][j] = myMat.at<uchar>(i, j);
    }
  }
  return data;
}

float own_median(std::vector<int> numbers)
{
  std::sort(numbers.begin(), numbers.end());
  int size = numbers.size();
  double median = 0;
  if (size % 2 == 0)
  {
    for(int i = 0; i<numbers.size() ; i++){
    }
    median = (numbers[size / 2 - 1] + numbers[size / 2]) / 2.0;
  }
  else
  {
    median = numbers[size / 2];
  }
  return median;
}

float own_median(std::array<int, 4> numbers)
{
  std::sort(numbers.begin(), numbers.end());
  int size = numbers.size();
  double median = 0;
  if (size % 2 == 0)
  {
    median = (numbers[size / 2 - 1] + numbers[size / 2]) / 2.0;
  }
  else
  {
    median = numbers[size / 2];
  }
  return median;
}

int get_w_st(int s, int t)
{
  switch (s)
  {
  case -1:
  case 0:
  case 1:
    if (t == -1 || t == 0 || t == 1)
    {
      return 2;
    }
    break;
  }
  return 1;
}

int calculate_y_plus_s(int y, int s, int height)
{
  if (y + s < 0)
  {
    return 0;
  }
  else if (y + s >= height)
  {
    return height - 1;
  }
  else
  {
    return y + s;
  }
}

int calculate_x_plus_t(int x, int t, int width)
{
  if (x + t < 0)
  {
    return 0;
  }
  else if (x + t >= width)
  {
    return width - 1;
  }
  else
  {
    return x + t;
  }
}

int own_argmin(std::vector<int> &arr)
{
  auto min_it = std::min_element(arr.begin(), arr.end());
  return std::distance(arr.begin(), min_it);
}

int own_argmin(std::array<int, 4>& arr)
{
  auto min_it = std::min_element(arr.begin(), arr.end());
  return std::distance(arr.begin(), min_it);
}

double own_mean(std::vector<int> &arr)
{
  if (arr.empty())
  {
    return 0;
  }
  double sum = 0;
  for (int x : arr)
  {
    sum += x;
  }
  return sum / arr.size();
}

double own_mean(std::array<int, 4>& arr)
{
  if (arr.empty())
  {
    return 0;
  }
  double sum = 0;
  for (int x : arr)
  {
    sum += x;
  }
  return sum / arr.size();
}

double own_std(std::vector<int> &arr)
{
  if (arr.empty())
  {
    return 0;
  }
  double mean = own_mean(arr);
  double variance = 0;
  for (int x : arr)
  {
    variance += (x - mean) * (x - mean);
  }
  variance /= arr.size();
  return std::sqrt(variance);
}

double own_std(std::array<int, 4>& arr)
{
  if (arr.empty())
  {
    return 0;
  }
  double mean = own_mean(arr);
  double variance = 0;
  for (int x : arr)
  {
    variance += (x - mean) * (x - mean);
  }
  variance /= arr.size();
  return std::sqrt(variance);
}

int is_s_t_in_coordinates(int s, int t, int l_ij, std::vector<std::vector<std::pair<int, int>>> &coordinates)
{
  for (const auto &direction : coordinates[l_ij])
  {
    if (direction.first == s && direction.second == t)
    {
      return 2;
    }
  }
  return 1;
}

const cv::Mat &add_noise_to_frame(cv::Mat &frame, float noise_percent)
{
  int pixels_to_be_noised = std::round(frame.total() * noise_percent);
  for (int count = 0; count < pixels_to_be_noised; count++)
  {
    int randomWidth = rand() % (frame.cols - 1);
    int randomHeight = rand() % (frame.rows - 1);
    frame.at<uchar>(randomHeight, randomWidth) = rand() % 255;
  }
  return frame;
}

cv::Mat generate_noise_map(cv::Mat _apx_of_noise)
{
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

std::tuple<cv::Rect, int, int> create_rect(int x, int y, int kernel, const cv::Mat &frame)
{
  int kernel_f = std::floor(kernel / 2);
  x = std::max(0, x - kernel_f);
  y = std::max(0, y - kernel_f);
  int max_x = std::min(frame.cols - x, kernel);
  int max_y = std::min(frame.rows - y, kernel);
  return std::make_tuple(cv::Rect(x, y, max_x, max_y), y, x);
}

double calc_weight(int middle_x, int middle_y, int rect_current_y, int rect_current_x, int current_frame, int rect_frame, int kernel_size)
{
  float distance = sqrt(pow(rect_current_x - middle_x, 2) + pow(rect_current_y - middle_y, 2) + pow(rect_frame - current_frame, 2));
  return exp(-distance / kernel_size);
}
// Calculate cubes
cv::Mat calculate_median_cube(std::vector<cv::Mat> &images, int actual_frame, int kernel)
{
  auto non_empty_mat = [](const cv::Mat &m)
  { return !m.empty(); };
  int count = std::count_if(images.begin(), images.end(), non_empty_mat);
  cv::Mat result = images[actual_frame].clone();
  cv::parallel_for_(cv::Range(0, images[0].rows), [&](const cv::Range &range)
                    {
      std::vector<int> values(kernel*kernel*count, 0);
      cv::Rect roi;
      cv::Mat submat;
      for (int y = range.start; y < range.end; y++) {
        for(int x = 0; x < images[0].cols; x++){
          int idx = 0;
          std::for_each(images.begin(), images.end(), [&](const cv::Mat& frame) {
            if (!frame.empty()) {
              roi = std::get<0>(create_rect(x, y, kernel, frame));
              submat = frame(roi);
              for (int i = 0; i < submat.rows; i++) {
                for (int j = 0; j < submat.cols; j++) {
                  values[idx++] = submat.at<uchar>(i,j);
                }
              }
            }
          });
          result.at<uchar>(y, x) = own_median(values);
          std::fill(values.begin(), values.end(), 0);
        }
      } });

  return result;
}

cv::Mat calculate_weighted_cube(std::vector<cv::Mat> &images, int actual_frame, int kernel, std::string &weight_type)
{
  auto non_empty_mat = [](const cv::Mat &m)
  { return !m.empty(); };
  int count = std::count_if(images.begin(), images.end(), non_empty_mat);
  cv::Mat result = images[actual_frame].clone();

  cv::parallel_for_(cv::Range(0, images[0].rows), [&](const cv::Range &range)
                    {
    cv::Mat hist_weights = cv::Mat::zeros(256, 1, CV_32FC1);
    cv::Rect roi;
    cv::Mat submat;
    for (int y = range.start; y < range.end; y++) {
    for(int x = 0; x < images[0].cols; x++){
      hist_weights *= 0;
      int frame_count = 0;
      std::for_each(images.begin(), images.end(), [&](const cv::Mat& frame) {
        if (!frame.empty()) {
          int r_x, r_y;
          std::tie(roi, r_y, r_x) = create_rect(x, y, kernel, frame);
          submat = frame(roi);
          for (int i = 0; i < submat.rows; i++) {
            for (int j = 0; j < submat.cols; j++) {
              double weight = calc_weight(x, y, r_y + i, r_x + j, actual_frame, frame_count, kernel);
              int value = submat.at<uchar>(i, j); 
              hist_weights.at<float>(value, 0) += weight;
            }
          }
        }
        frame_count++;
      });
      float median = 0;
      float sum = 0;
      float middle = kernel * kernel * count * 0.25;
      for (int i = 0; i < 256; i++) {
          sum += hist_weights.at<float>(i, 0);
          if (sum >= middle) {
                median = i;
                break;
          }
      }
      result.at<uchar>(y, x) = median;
    }
  } });
  return result;
}

cv::Mat calculate_two_pass_cube(std::vector<cv::Mat> &images, int actual_frame)
{
  auto non_empty_mat = [](const cv::Mat &m)
  { return !m.empty(); };
  int count = std::count_if(images.begin(), images.end(), non_empty_mat);
  cv::Mat apx_of_signal_only;
  cv::medianBlur(images[actual_frame], apx_of_signal_only, 3);
  cv::Mat apx_of_noise;
  cv::absdiff(images[actual_frame], apx_of_signal_only, apx_of_noise);
  cv::Mat noise_map = generate_noise_map(apx_of_noise);
  cv::Mat result = images[actual_frame];
  std::vector<int> values((count - 1), 0);
  cv::Rect roi;
  cv::Mat submat;
  int kernel = 3;
  std::vector<int> vec_values(count * kernel * kernel, 0);
  for (int y = 0; y < images[0].rows; y++)
  {
    for (int x = 0; x < images[0].cols; x++)
    {
      uchar noise = noise_map.at<uchar>(y, x);
      uchar pixel = images[actual_frame].at<uchar>(y, x);
      if (noise_map.at<uchar>(y, x) == 0)
      {
        continue;
      }
      int id = 0;
      std::for_each(images.begin(), images.end(), [&](const cv::Mat &frame)
                    {
        if (!frame.empty()) {
          values[id] = frame.at<uchar>(y,x);
        } });
      int median = std::round(own_median(values));
      if (median == 0)
      {
        int idx = 0;
        std::for_each(images.begin(), images.end(), [&](const cv::Mat &frame)
                      {
          if (!frame.empty()) {
            roi = std::get<0>(create_rect(x, y, kernel, frame));
            submat = frame(roi);
            for (int i = 0; i < submat.rows; i++) {
              for (int j = 0; j < submat.cols; j++) {
                vec_values[idx++] = submat.at<uchar>(i,j);
              }
            }
          } });
        median = own_median(vec_values);
        std::fill(vec_values.begin(), vec_values.end(), 0);
      }
      result.at<uchar>(y, x) = median;
    }
  }
  return result;
}

cv::Mat calculate_dir_w_cube(std::vector<cv::Mat> &images, int actual_frame, int threshold)
{
  auto non_empty_mat = [](const cv::Mat &m)
  { return !m.empty(); };
  int count = std::count_if(images.begin(), images.end(), non_empty_mat);
  std::vector<std::vector<std::pair<int, int>>> coordinates = {
      {{-2, -2}, {-1, -1}, {1, 1}, {2, 2}}, // S_1
      {{0, -2}, {0, -1}, {0, 1}, {0, 2}},   // S_2
      {{2, -2}, {1, -1}, {-1, 1}, {-2, 2}}, // S_3
      {{-2, 0}, {-1, 0}, {1, 0}, {2, 0}},   // S_4
  };
  std::vector<std::vector<std::pair<int, int>>> o_3 = {
      {{-1, 1}, {1, 1}},
      {{0, -1}, {0, 1}},
      {{1, -1}, {-1, 1}},
      {{-1, 0}, {1, 0}}};

  int height = images[0].rows;
  int width = images[0].cols;
  cv::Mat u_ij = cv::Mat::zeros(height, width, CV_8UC1);
  cv::parallel_for_(cv::Range(0, images[0].rows), [&](const cv::Range &range){
    for (int y = range.start; y < range.end; y++)
    {
      std::array<int, 4> d_k{};
      std::array<int, 4> std_k{};
      std::array<int, 4> dir_std{};
      std::vector<int> values_for_pixel;
      for (int x = 0; x < width; x++)
      {
        std::fill(d_k.begin(), d_k.end(), 0);
        std::fill(std_k.begin(), std_k.end(), 0);
        for (int direction = 0; direction < 4; direction++)
        {
          int d_sum = 0;
          std::fill(dir_std.begin(), dir_std.end(), 0);
          int counter = 0;
          for (const auto &point : coordinates[direction])
          {
            int s = point.first;
            int t = point.second;
            int w_st = get_w_st(s, t);
            int y_plus_s = calculate_y_plus_s(y, s, height);
            int x_plus_t = calculate_x_plus_t(x, t, width);
            int w_st_times_abs_y_with_st_minus_y = w_st * std::abs(images[actual_frame].at<uchar>(y_plus_s, x_plus_t) - images[actual_frame].at<uchar>(y, x));
            d_sum += w_st_times_abs_y_with_st_minus_y;
            dir_std[counter] = images[actual_frame].at<uchar>(y_plus_s, x_plus_t);
            counter++;
          }
          d_k[direction] = d_sum;
          std_k[direction] = own_std(dir_std);
        }
        int r_ij = *std::min_element(d_k.begin(), d_k.end());
        int l_ij = own_argmin(std_k);
        if (r_ij > threshold){
          std::for_each(images.begin(), images.end(), [&](const cv::Mat &frame){
            if(!frame.empty()){
              for (int direction = 0; direction < 4; direction++)
              {
                for (const auto &point : o_3[direction])
                {
                  int s = point.first;
                  int t = point.second;
                  int w_st = is_s_t_in_coordinates(s, t, l_ij, coordinates);
                  int y_plus_s = calculate_y_plus_s(y, s, height);
                  int x_plus_t = calculate_x_plus_t(x, t, width);
                  for (int rep = 0; rep < w_st; rep++)
                  {
                    values_for_pixel.push_back(frame.at<uchar>(y_plus_s, x_plus_t));
                  }
                }
              }
            }
          });
          u_ij.at<uchar>(y, x) = own_median(values_for_pixel);
          values_for_pixel.clear();
        }else{
          u_ij.at<uchar>(y,x) = images[actual_frame].at<uchar>(y,x);
        } 
      }
    }
  });
  std::cout << "frame" << std::endl;
  return u_ij;
}

// Image filters--------------------------------------------------------------------------------------
cv::Mat basic_median_mat(cv::Mat &n_image, int kernel_size)
{
  cv::Mat filtered;
  cv::medianBlur(n_image, filtered, kernel_size);
  return filtered;
}

cv::Mat directional_weighted_median_mat(cv::Mat &n_image, int threshold, int height, int width)
{
  std::vector<std::vector<std::pair<int, int>>> coordinates = {
      {{-2, -2}, {-1, -1}, {1, 1}, {2, 2}}, // S_1
      {{0, -2}, {0, -1}, {0, 1}, {0, 2}},   // S_2
      {{2, -2}, {1, -1}, {-1, 1}, {-2, 2}}, // S_3
      {{-2, 0}, {-1, 0}, {1, 0}, {2, 0}},   // S_4
  };
  std::vector<std::vector<std::pair<int, int>>> o_3 = {
      {{-1, 1}, {1, 1}},
      {{0, -1}, {0, 1}},
      {{1, -1}, {-1, 1}},
      {{-1, 0}, {1, 0}}};
  std::array<int, 4> d_k{};
  std::array<int, 4> std_k{};
  std::array<int, 4> dir_std{};
  std::vector<int> values_for_pixel;
  cv::Mat u_ij = cv::Mat::zeros(height, width, CV_8UC1);
  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      std::fill(d_k.begin(), d_k.end(), 0);
      std::fill(std_k.begin(), std_k.end(), 0);
      for (int direction = 0; direction < 4; direction++)
      {
        int d_sum = 0;
        std::fill(dir_std.begin(), dir_std.end(), 0);
        int counter = 0;
        for (const auto &point : coordinates[direction])
        {
          int s = point.first;
          int t = point.second;
          int w_st = get_w_st(s, t);
          int y_plus_s = calculate_y_plus_s(y, s, height);
          int x_plus_t = calculate_x_plus_t(x, t, width);
          int w_st_times_abs_y_with_st_minus_y = w_st * std::abs(n_image.at<uchar>(y_plus_s, x_plus_t) - n_image.at<uchar>(y, x));
          d_sum += w_st_times_abs_y_with_st_minus_y;
          dir_std[counter] = n_image.at<uchar>(y_plus_s, x_plus_t);
          counter++;
        }
        d_k[direction] = d_sum;
        std_k[direction] = own_std(dir_std);
      }

      int r_ij = *std::min_element(d_k.begin(), d_k.end());
      int l_ij = own_argmin(std_k);

      int alpha_ij = 0;
      if (r_ij <= threshold)
      {
        u_ij.at<uchar>(y, x) = n_image.at<uchar>(y, x);
        continue;
      }

      values_for_pixel.clear();
      for (int direction = 0; direction < 4; direction++)
      {
        for (const auto &point : o_3[direction])
        {
          int s = point.first;
          int t = point.second;
          int w_st = is_s_t_in_coordinates(s, t, l_ij, coordinates);
          int y_plus_s = calculate_y_plus_s(y, s, height);
          int x_plus_t = calculate_x_plus_t(x, t, width);
          for (int rep = 0; rep < w_st; rep++)
          {
            values_for_pixel.push_back(n_image.at<uchar>(y_plus_s, x_plus_t));
          }
        }
      }
      u_ij.at<uchar>(y, x) = own_median(values_for_pixel);
    }
  }
  return u_ij;
}

cv::Mat weighted_median_filter_mat(cv::Mat &image, int kernel_size, std::string weight_type = "uniform")
{
  int padding = kernel_size / 2;
  cv::Mat padded_image;
  cv::copyMakeBorder(image, padded_image, padding, padding, padding, padding, cv::BORDER_REPLICATE);
  cv::Mat weights;
  if (weight_type == "uniform")
  {
    weights = cv::Mat::ones(kernel_size, kernel_size, CV_32FC1);
  }
  else if (weight_type == "distance")
  {
    int center = (kernel_size - 1) / 2;
    weights = cv::Mat::zeros(kernel_size, kernel_size, CV_32FC1);
    for (int x = 0; x < kernel_size; x++)
    {
      for (int y = 0; y < kernel_size; y++)
      {
        float distance = sqrt(pow(x - center, 2) + pow(y - center, 2));
        weights.at<float>(y, x) = exp(-distance / kernel_size);
      }
    }
  }
  else
  {
    throw std::invalid_argument("Invalid weight type. Use 'uniform' or 'distance'.");
  }
  cv::Mat output_image(image.rows, image.cols, image.type());
  std::mutex output_image_mutex;

  auto process_neighborhood = [&](const cv::Range &range)
  {
    cv::Mat hist_weights = cv::Mat::zeros(256, 1, CV_32FC1);
    for (int y = range.start; y < range.end; y++)
    {
      for (int x = 0; x < image.cols; x++)
      {
        cv::Rect roi(x, y, kernel_size, kernel_size);
        cv::Mat neighborhood = padded_image(roi);
        hist_weights *= 0;
        for (int i = 0; i < neighborhood.rows; i++)
        {
          for (int j = 0; j < neighborhood.cols; j++)
          {
            float weight = weights.at<float>(i, j);
            int value = neighborhood.at<uchar>(i, j);
            hist_weights.at<float>(value, 0) += weight;
          }
        }
        float median = 0;
        float sum = 0;
        float middle = kernel_size * kernel_size * 0.5;
        for (int i = 0; i < 256; i++)
        {
          sum += hist_weights.at<float>(i, 0);
          if (sum >= middle)
          {
            median = i;
            break;
          }
        }
        std::lock_guard<std::mutex> lock(output_image_mutex);
        output_image.at<uchar>(y, x) = median;
      }
    }
  };

  int num_threads = cv::getNumberOfCPUs();
  cv::parallel_for_(cv::Range(0, image.rows), process_neighborhood, num_threads);
  return output_image;
}

cv::Mat two_pass_median_for_image_mat(const cv::Mat &image_to_filter)
{
  cv::Mat apx_of_signal_only;
  cv::medianBlur(image_to_filter, apx_of_signal_only, 3);
  cv::Mat apx_of_noise;
  cv::absdiff(image_to_filter, apx_of_signal_only, apx_of_noise);
  cv::Mat noise_map = generate_noise_map(apx_of_noise);
  cv::Mat _filtered_image = image_to_filter.clone();
  apx_of_signal_only.copyTo(_filtered_image, noise_map);
  return _filtered_image;
}
// Video filters-----------------------------------------------------------------------------------------------------------
double simple_median_for_video_frame(const std::string &video_path, const std::string &video_name, int kernel_size, std::string output_path = "")
{
  cv::VideoCapture capture(video_path + video_name);
  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
    return 0.0;
  }
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
  std::string output = "";
  if(output_path == ""){
    output = video_path + "simple_median_" + std::to_string(kernel_size) + "_" + video_name;
  }else{
    output = output_path + "simple_median_" + std::to_string(kernel_size) + "_" + video_name;
  }

  cv::VideoWriter writer(output, fourcc, fps, cv::Size(width, height), 0);
  double psnr_sum = 0.0;

  cv::Mat frame;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    cv::medianBlur(frame, blurred, kernel_size);
    psnr_sum += PSNR(frame, blurred);
    writer.write(blurred);
    if (cv::waitKey(33) >= 0)
    {
      break;
    }
  }
  capture.release();
  writer.release();
  std::cout << "done" << std::endl;
  return psnr_sum / frame_count;
}

double weighted_median_for_video_frame(const std::string &video_path, const std::string &video_name, int kernel_size, std::string &weight_type, std::string output_path = "")
{
  cv::VideoCapture capture(video_path + video_name);
  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
    return 0.0;
  }
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
  std::string output = "";
  if(output_path == ""){
    output = video_path + "weighted_median_" + weight_type + "_" + std::to_string(kernel_size) + "_" + video_name;
  }else{
    output = output_path + "weighted_median_"  + weight_type + "_" + std::to_string(kernel_size) + "_" + video_name;
  }
  cv::VideoWriter writer(output, fourcc, fps, cv::Size(width, height), 0);  double psnr_sum = 0.0;
  cv::Mat frame;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    blurred = weighted_median_filter_mat(frame, kernel_size, weight_type);
    psnr_sum += PSNR(frame, blurred);
    writer.write(blurred);
    if (cv::waitKey(33) >= 0)
    {
      break;
    }
  }
  capture.release();
  writer.release();
  return psnr_sum / frame_count;
}

double directional_weighted_median_for_video_frame(const std::string &video_path, const std::string &video_name, int threshold, std::string output_path = "")
{
  cv::VideoCapture capture(video_path + video_name);
  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
    return 0.0;
  }
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
  std::string output = "";
  if(output_path == ""){
    output = video_path + "dir_w_median_" + std::to_string(threshold) + "_" + video_name;
  }else{
    output = output_path + "dir_w_median_"+ std::to_string(threshold) + "_" + video_name;
  }
  cv::VideoWriter writer(output, fourcc, fps, cv::Size(width, height), 0);  double psnr_sum = 0.0;
  cv::Mat frame;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    blurred = directional_weighted_median_mat(frame, threshold, height, width);
    psnr_sum += PSNR(frame, blurred);
    writer.write(blurred);
    if (cv::waitKey(33) >= 0)
    {
      break;
    }
  }
  capture.release();
  writer.release();
  return psnr_sum / frame_count;
}

double two_pass_median_median_for_video_frame(const std::string &video_path, const std::string &video_name, std::string output_path = "")
{
  cv::VideoCapture capture(video_path + video_name);
  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
    return 0.0;
  }
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
  std::string output = "";
  if(output_path == ""){
    output = video_path + "two_pass_median_" + video_name;
  }else{
    output = output_path + "two_pass_median_" + video_name;
  }
  cv::VideoWriter writer(output, fourcc, fps, cv::Size(width, height), 0);
  double psnr_sum = 0.0;


  cv::Mat frame;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::Mat blurred;
    blurred = two_pass_median_for_image_mat(frame);
    psnr_sum += PSNR(frame, blurred);
    writer.write(blurred);
    if (cv::waitKey(33) >= 0)
    {
      break;
    }
  }
  capture.release();
  writer.release();
  return psnr_sum / frame_count;
}
// Callers to python---------------------------------------------------------------------------------------------------
void add_noise_to_video(const std::string &video_path, const std::string &video_name, float noise_percent, std::string output_path = "")
{
  cv::VideoCapture capture(video_path + video_name);
  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
    return;
  }
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  std::string output = "";
  if(output_path == ""){
    output = video_path + "_" + std::to_string(noise_percent * 100) + "_noise_" + video_name;
  }else{
    output = output_path + "_" + std::to_string(noise_percent * 100) + "_noise_" + video_name;
  }
  cv::VideoWriter writer(output, fourcc, fps, cv::Size(width, height), 0);
  cv::Mat frame;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frame = add_noise_to_frame(frame, noise_percent);
    writer.write(frame);
    if (cv::waitKey(33) >= 0)
    {
      break;
    }
  }
  capture.release();
  writer.release();
}

std::tuple<std::vector<std::vector<int>>, double> weighted_median_filter_vector(std::vector<std::vector<int>> n_image, int kernel_size, std::string weight_type = "uniform")
{
  cv::Mat myMat = convert_vector_to_mat(n_image);
  cv::Mat myMat_f = weighted_median_filter_mat(myMat, kernel_size, weight_type);
  return std::make_tuple(convert_mat_to_vector(myMat_f), PSNR(myMat, myMat_f));
}

std::tuple<std::vector<std::vector<int>>, double> directional_weighted_median_vector(std::vector<std::vector<int>> n_image, int threshold, int height, int width)
{
  cv::Mat myMat = convert_vector_to_mat(n_image);
  cv::Mat myMat_f = directional_weighted_median_mat(myMat, threshold, height, width);
  return std::make_tuple(convert_mat_to_vector(myMat_f), PSNR(myMat, myMat_f));
}

std::tuple<std::vector<std::vector<int>>, double> two_pass_median_for_image_vector(std::vector<std::vector<int>> n_image)
{
  cv::Mat myMat = convert_vector_to_mat(n_image);
  cv::Mat myMat_f = two_pass_median_for_image_mat(myMat);
  return std::make_tuple(convert_mat_to_vector(myMat_f), PSNR(myMat, myMat_f));
}

std::tuple<std::vector<std::vector<int>>, double> basic_median_for_image_vector(std::vector<std::vector<int>> n_image, int kernel_size)
{
  cv::Mat myMat = convert_vector_to_mat(n_image);
  cv::Mat myMat_f = basic_median_mat(myMat, kernel_size);
  return std::make_tuple(convert_mat_to_vector(myMat_f), PSNR(myMat, myMat_f));
}

// Video methods with neighbors-----------------------------------------------------------------------------------------
double simple_median_cube(std::string &video_path, std::string &video_name, int kernel_size, int neighbors, std::string output_path = "")
{
  cv::VideoCapture capture(video_path + video_name);
  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
    return 0.0;
  }
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
  std::string output = "";
  if(output_path == ""){
    output = video_path + "simple_cube_" + std::to_string(neighbors) + "_" + std::to_string(kernel_size) + "_" + video_name;
  }else{
    output = output_path + "simple_cube_" + std::to_string(neighbors) + "_" + std::to_string(kernel_size) + "_" + video_name;
  }
  cv::VideoWriter writer(output, fourcc, fps, cv::Size(width, height), 0);  double psnr_sum = 0.0;
  cv::Mat frame;
  int counter = -neighbors;
  std::vector<cv::Mat> neighbors_vec(1 + 2 * neighbors, cv::Mat());
  cv::Mat blurred;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    if (counter < 0)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      counter++;
    }
    else if (counter < neighbors)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_median_cube(neighbors_vec, counter, kernel_size);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
    else if (counter == neighbors)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_median_cube(neighbors_vec, counter, kernel_size);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
    else
    {
      counter = neighbors;
      std::rotate(neighbors_vec.begin(), neighbors_vec.begin() + 1, neighbors_vec.end());
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_median_cube(neighbors_vec, counter, kernel_size);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
  }
  while (counter < neighbors)
  {
    neighbors_vec[counter - neighbors] = cv::Mat();
    blurred = calculate_median_cube(neighbors_vec, counter, kernel_size);
    psnr_sum += PSNR(neighbors_vec[counter], blurred);
    writer.write(blurred);
    counter++;
  }
  capture.release();
  writer.release();
  return psnr_sum / frame_count;
}

double weighted_median_cube(std::string &video_path, std::string &video_name, int kernel_size, std::string &weight_type, int neighbors, std::string output_path = "")
{
  cv::VideoCapture capture(video_path + video_name);
  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
    return 0.0;
  }
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
  std::string output = "";
  if(output_path == ""){
    output = video_path + "weighted_median_" + std::to_string(neighbors) + "_" + std::to_string(kernel_size) + "_" + weight_type + "_" + video_name;
  }else{
    output = output_path + "weighted_median_" + std::to_string(neighbors) + "_" + std::to_string(kernel_size) + "_" + weight_type + "_" + video_name;
  }
  cv::VideoWriter writer(output, fourcc, fps, cv::Size(width, height), 0);  double psnr_sum = 0.0;
  cv::Mat frame;
  int counter = -neighbors;
  std::vector<cv::Mat> neighbors_vec(1 + 2 * neighbors, cv::Mat());
  cv::Mat blurred;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    if (counter < 0)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      counter++;
    }
    else if (counter < neighbors)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_weighted_cube(neighbors_vec, counter, kernel_size, weight_type);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
    else if (counter == neighbors)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_weighted_cube(neighbors_vec, counter, kernel_size, weight_type);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
    else
    {
      counter = neighbors;
      std::rotate(neighbors_vec.begin(), neighbors_vec.begin() + 1, neighbors_vec.end());
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_weighted_cube(neighbors_vec, counter, kernel_size, weight_type);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
  }
  while (counter < neighbors)
  {
    neighbors_vec[counter - neighbors] = cv::Mat();
    blurred = calculate_weighted_cube(neighbors_vec, counter, kernel_size, weight_type);
    psnr_sum += PSNR(neighbors_vec[counter], blurred);
    writer.write(blurred);
    counter++;
  }
  capture.release();
  writer.release();
  return psnr_sum / frame_count;
}

double two_pass_median_cube(std::string &video_path, std::string &video_name, int neighbors, std::string output_path = "")
{
  cv::VideoCapture capture(video_path + video_name);
  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
    return 0.0;
  }
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
  std::string output = "";
  if(output_path == ""){
    output = video_path + "two_pass_median_" + std::to_string(neighbors) + "_" + video_name;
  }else{
    output = output_path + "two_pass_median_" + std::to_string(neighbors) + "_" + video_name;
  }
  cv::VideoWriter writer(output, fourcc, fps, cv::Size(width, height), 0);  double psnr_sum = 0.0;
  cv::Mat frame;
  int counter = -neighbors;
  std::vector<cv::Mat> neighbors_vec(1 + 2 * neighbors, cv::Mat());
  cv::Mat blurred;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    if (counter < 0)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      counter++;
    }
    else if (counter < neighbors)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_two_pass_cube(neighbors_vec, counter);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
    else if (counter == neighbors)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_two_pass_cube(neighbors_vec, counter);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
    else
    {
      counter = neighbors;
      std::rotate(neighbors_vec.begin(), neighbors_vec.begin() + 1, neighbors_vec.end());
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_two_pass_cube(neighbors_vec, counter);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
  }
  while (counter < neighbors)
  {
    neighbors_vec[counter - neighbors] = cv::Mat();
    blurred = calculate_two_pass_cube(neighbors_vec, counter);
    psnr_sum += PSNR(neighbors_vec[counter], blurred);
    writer.write(blurred);
    counter++;
  }
  capture.release();
  writer.release();
  return psnr_sum / frame_count;
}

double dir_w_median_cube(std::string &video_path, std::string &video_name, int threshold, int neighbors, std::string output_path = "")
{
  cv::VideoCapture capture(video_path + video_name);
  if (!capture.isOpened())
  {
    std::cerr << "Unable to open video file: " << video_path << std::endl;
    return 0.0;
  }
  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fps = capture.get(cv::CAP_PROP_FPS);
  int fourcc = capture.get(cv::CAP_PROP_FOURCC);
  int frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
  std::string output = "";
  if(output_path == ""){
    output = video_path + "dir_w_cube_" + std::to_string(neighbors) + "_" + std::to_string(threshold) + "_" + video_name;
  }else{
    output = output_path + "dir_w_cube_" + std::to_string(neighbors) + "_" + std::to_string(threshold) + "_" + video_name;
  }
  cv::VideoWriter writer(output, fourcc, fps, cv::Size(width, height), 0);  double psnr_sum = 0.0;
  cv::Mat frame;
  int counter = -neighbors;
  std::vector<cv::Mat> neighbors_vec(1 + 2 * neighbors, cv::Mat());
  cv::Mat blurred;
  while (capture.read(frame))
  {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    if (counter < 0)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      counter++;
    }
    else if (counter < neighbors)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_dir_w_cube(neighbors_vec, counter, threshold);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
    else if (counter == neighbors)
    {
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_dir_w_cube(neighbors_vec, counter, threshold);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
    else
    {
      counter = neighbors;
      std::rotate(neighbors_vec.begin(), neighbors_vec.begin() + 1, neighbors_vec.end());
      neighbors_vec[counter + neighbors] = frame.clone();
      blurred = calculate_dir_w_cube(neighbors_vec, counter, threshold);
      psnr_sum += PSNR(frame, blurred);
      writer.write(blurred);
      counter++;
    }
  }
  while (counter < neighbors)
  {
    neighbors_vec[counter - neighbors] = cv::Mat();
    blurred = calculate_dir_w_cube(neighbors_vec, counter, threshold);
    psnr_sum += PSNR(neighbors_vec[counter], blurred);
    writer.write(blurred);
    counter++;
  }
  capture.release();
  writer.release();
  return psnr_sum / frame_count;
}

PYBIND11_MODULE(cpp_calculate, module_handle)
{
  module_handle.doc() = "I'm a docstring hehe";
  module_handle.def("add_noise_to_video", &add_noise_to_video);
  module_handle.def("downscale_video_res", &downscale_video_res);

  module_handle.def("directional_weighted_median_vector", &directional_weighted_median_vector);
  module_handle.def("two_pass_median_for_image_vector", &two_pass_median_for_image_vector);
  module_handle.def("weighted_median_filter_vector", &weighted_median_filter_vector);
  module_handle.def("basic_median_for_image_vector", &basic_median_for_image_vector);

  module_handle.def("simple_median_for_video_frame", &simple_median_for_video_frame);
  module_handle.def("weighted_median_for_video_frame", &weighted_median_for_video_frame);
  module_handle.def("directional_weighted_median_for_video_frame", &directional_weighted_median_for_video_frame);
  module_handle.def("two_pass_median_for_video_frame", &two_pass_median_median_for_video_frame);

  module_handle.def("simple_median_cube", &simple_median_cube);
  module_handle.def("weighted_median_cube", &weighted_median_cube);
  module_handle.def("two_pass_median_cube", &two_pass_median_cube);
  module_handle.def("dir_w_median_cube", &dir_w_median_cube);
}