#include "common/image_utils.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

namespace image_utils {

std::vector<cv::Vec3f> loadImage(const string &image_path,
                                 cv::Mat &img_float, int &num_rows) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Image Empty\n";
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    num_rows = image.rows;

    // Reshape to 2D array, one row is each pixel
    cv::Mat image_float;
    image.convertTo(image_float, CV_32F);
    image_float = image_float.reshape(1, image.rows * image.cols);

    vector<cv::Vec3f> pixels;
    pixels.reserve(image_float.rows);

    for (int i = 0; i < image_float.rows; i++) {
        cv::Vec3f pixel = image_float.at<cv::Vec3f>(i, 0);
        pixels.push_back(pixel);
    }
    img_float = image_float;
    return pixels;
}

} // namespace image_utils
