#include "common/image_utils.hpp"
#include "common/constants.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

namespace image_utils {

vector<cv::Vec3f> loadImage(const string &image_path,
                                 cv::Mat &img_float, int &num_rows) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        cerr << "Image Empty\n";
        throw runtime_error("Failed to load image: " + image_path);
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

cv::Mat load_float_image(const std::string& input) {
    cv::Mat img = cv::imread(input);
    if (img.empty()) {
        std::cerr << "Failed to load image from path: " << input << "\n";
    }
    img.convertTo(img, CV_32F, 1.0F / MAX_PIXEL_VALUE);
    return img;
}

std::vector<float> flatten_image(const cv::Mat& img, size_t N) {
    // Flatten pixel data: row-major, channels interleaved
    std::vector<float> h_pixels(N * PIXEL_DIM);
    for (size_t y = 0, idx = 0; y < static_cast<size_t>(img.rows); ++y) {
        for (size_t x = 0; x < static_cast<size_t>(img.cols); ++x, ++idx) {
            const auto& pix
                = img.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x));
            for (size_t i = 0; i < PIXEL_DIM; i++) {
                h_pixels[(idx * PIXEL_DIM) + i] = pix[static_cast<int>(i)];
            }
        }
    }
    return h_pixels;
}

void produce_image(size_t rows, size_t cols, const std::vector<int>& h_labels,
                   const std::vector<float>& h_centroids,
                   const std::string& out_path) {
    cv::Mat out((int)rows, (int)cols, CV_32FC3);
    for (size_t i = 0, idx = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j, ++idx) {
            int k = h_labels[idx];
            out.at<cv::Vec3f>(static_cast<int>(i), static_cast<int>(j))
                = cv::Vec3f(h_centroids[(k * PIXEL_DIM) + 0],
                            h_centroids[(k * PIXEL_DIM) + 1],
                            h_centroids[(k * PIXEL_DIM) + 2]);
        }
    }
    out.convertTo(out, CV_8UC3, MAX_PIXEL_VALUE);
    cv::imwrite(out_path, out);
}

} // namespace image_utils
