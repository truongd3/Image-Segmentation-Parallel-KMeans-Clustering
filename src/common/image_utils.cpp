#include "common/image_utils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

namespace ImageUtils {

std::vector<cv::Vec3f> loadImage(const std::string &image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cout << "Image Empty\n";
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    std::vector<cv::Vec3f> pixels;
    pixels.reserve((size_t)image.rows * image.cols);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3f color = image.at<cv::Vec3f>(i, j);
            pixels.push_back(color);
        }
    }
    return pixels;
}

} // namespace ImageUtils
