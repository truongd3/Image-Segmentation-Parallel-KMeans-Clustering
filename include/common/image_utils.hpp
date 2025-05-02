#pragma once

#include <opencv2/opencv.hpp>

namespace image_utils {
std::vector<cv::Vec3f> loadImage(const std::string& image_path,
                                 cv::Mat& img_float, int& num_rows);

cv::Mat load_float_image(const std::string& input);

std::vector<float> flatten_image(const cv::Mat& img, size_t N);

void produce_image(size_t rows, size_t cols, const std::vector<int>& h_labels,
                   const std::vector<float>& h_centroids,
                   const std::string& out_path);

} // namespace image_utils
