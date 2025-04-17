#pragma once

#include <opencv2/opencv.hpp>

namespace ImageUtils {
std::vector<cv::Vec3f> loadImage(const std::string &image_path);
}
