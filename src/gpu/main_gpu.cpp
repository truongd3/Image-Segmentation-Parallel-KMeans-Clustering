#include "common/constants.hpp"
#include "common/image_utils.hpp"
#include "gpu/img_seg_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <image_path> <k> <output_path>\n";
        return 1;
    }

    string image_path = argv[1];
    size_t k = stoi(argv[2]);
    string output_path = argv[3];

    cv::Mat img = image_utils::load_float_image(image_path);
    size_t rows = img.rows;
    size_t cols = img.cols;
    size_t N = rows * cols;

    vector<float> h_pixels = image_utils::flatten_image(img, N);
    vector<float> h_centroids(k * PIXEL_DIM);

    vector<int> h_labels(N);

    cudaEvent_t start_time{};
    cudaEvent_t end_time{};
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    cudaEventRecord(start_time);
    gpu::img_seg_gpu(k, N, h_pixels, h_centroids, h_labels);
    image_utils::produce_image(rows, cols, h_labels, h_centroids, output_path);
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);

    float ms = 0.0F;
    cudaEventElapsedTime(&ms, start_time, end_time);
    double secs = ms / 1000.0;
    cout << "CUDA version -> GPU kernel time: " << secs << " seconds\n";

    return 0;
}
