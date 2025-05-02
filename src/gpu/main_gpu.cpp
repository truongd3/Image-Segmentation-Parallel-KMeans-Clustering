#include "common/constants.hpp"
#include "common/image_utils.hpp"
#include "gpu/img_seg_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <k> <output_path>"
                  << '\n';
        return 1;
    }
    std::string image_path = argv[1];
    size_t k = std::stoi(argv[2]);
    std::string output_path = argv[3];

    cv::Mat img = image_utils::load_float_image(image_path);
    size_t rows = img.rows;
    size_t cols = img.cols;
    size_t N = rows * cols;

    std::vector<float> h_pixels = image_utils::flatten_image(img, N);
    std::vector<float> h_centroids(k * PIXEL_DIM);

    

    std::vector<int> h_labels(N);

    gpu::img_seg_gpu(k, N, h_pixels, h_centroids, h_labels);
    image_utils::produce_image(rows, cols, h_labels, h_centroids, output_path);

    return 0;
}
