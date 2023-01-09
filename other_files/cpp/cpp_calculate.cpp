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


namespace py = pybind11;

void add_noise_to_video(const cv::Mat &image){
    cv::imshow("Image", image);
    cv::waitKey(0);
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
}