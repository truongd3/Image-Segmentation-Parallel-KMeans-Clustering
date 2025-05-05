#include "common/constants.hpp"
#include "common/kmeans_utils.hpp"
#include "gpu/kmeans_kernels.cuh"
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <vector>

using namespace std;

#define CHECK_CUDA(err)                                                        \
    if (err != cudaSuccess) {                                                  \
        cerr << "CUDA error " << cudaGetErrorString(err) << " at "             \
                  << __LINE__ << '\n';                                         \
        exit(1);                                                               \
    }

namespace gpu {
void img_seg_gpu(size_t K, size_t N, const vector<float>& h_pixels,
                 vector<float>& h_centroids, vector<int>& h_labels) {

    kmeans_utils::init_centroids(h_pixels, h_centroids, N, K);

    float* d_pixels{};
    float* d_centroids{};
    float* d_sums{};
    int* d_labels{};
    int* d_counts{};

    const size_t d_pixels_size = N * PIXEL_DIM * sizeof(float);
    const size_t d_centroids_size = K * PIXEL_DIM * sizeof(float);
    const size_t d_labels_size = N * sizeof(int);
    const size_t d_sums_size = K * PIXEL_DIM * sizeof(float);
    const size_t d_counts_size = K * sizeof(int);

    CHECK_CUDA(cudaMalloc(&d_pixels, d_pixels_size));
    CHECK_CUDA(cudaMalloc(&d_centroids, d_centroids_size));
    CHECK_CUDA(cudaMalloc(&d_labels, d_labels_size));
    CHECK_CUDA(cudaMalloc(&d_sums, d_sums_size));
    CHECK_CUDA(cudaMalloc(&d_counts, d_counts_size));

    CHECK_CUDA(cudaMemcpy(d_pixels, h_pixels.data(), d_pixels_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), d_centroids_size, cudaMemcpyHostToDevice));

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    vector<float> h_sums(K * PIXEL_DIM);
    vector<int> h_counts(K);
    for (int iter = 0; iter < MAX_ITERS; iter++) {
        assign_clusters<<<gridDim, blockDim>>>(d_labels, d_pixels, d_centroids, N, K);

        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_sums, 0, d_sums_size));
        CHECK_CUDA(cudaMemset(d_counts, 0, d_counts_size));

        accumulate_clusters<<<gridDim, blockDim>>>(d_pixels, d_labels, d_sums, d_counts, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_sums.data(), d_sums, d_sums_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts, d_counts_size, cudaMemcpyDeviceToHost));

        // bool converged = true;
        for (int clus = 0; clus < K; clus++) {
            if (h_counts[clus] == 0) continue;
            for (int d = 0; d < PIXEL_DIM; d++) {
                float new_clus_comp = h_sums[(clus * PIXEL_DIM) + d] / (float)h_counts[clus];
                // if (abs(new_clus_comp - h_centroids[(clus * PIXEL_DIM) + d]) > TOL) converged = false;
                h_centroids[(clus * PIXEL_DIM) + d] = new_clus_comp;
            }
        }

        CHECK_CUDA(cudaMemcpy(d_centroids, h_centroids.data(), d_centroids_size, cudaMemcpyHostToDevice));

        // if (converged) {
        //     cout << "Converged at iteration " << iter << "\n";
        //     break;
        // }
    }
    CHECK_CUDA(cudaMemcpy(h_labels.data(), d_labels, d_labels_size, cudaMemcpyDeviceToHost));

    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);
}
} // namespace gpu
