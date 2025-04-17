#pragma once

#include <opencv2/opencv.hpp>

namespace image_utils {
std::vector<cv::Vec3f> loadImage(const std::string &image_path,
                                 cv::Mat &img_float, int &num_rows);
}
