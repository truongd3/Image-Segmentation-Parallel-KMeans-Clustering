#include "common/constants.hpp"
#include "gpu/kmeans_kernels.cuh"
#include <cfloat>

/**
 * @brief CUDA kernel to assign cluster for each data point
 * @param centroids array of centroids
 * @param labels store resulting cluster of each data point
 * @param N size of data points
 * @param K number of centroids
 */
__global__ void assign_clusters(int* labels, const float* pixels,
                                const float* centroids, size_t N, size_t K) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= N) {
        return;
    }

    float best_d = FLT_MAX;
    int best_k = 0;

    const float* pix = pixels + (idx * PIXEL_DIM);
    for (int c = 0; c < K; ++c) {
        const float* cent = centroids + (size_t)(c * PIXEL_DIM);
        float dist = 0;
        for (int d = 0; d < PIXEL_DIM; d++) {
            float diff = pix[d] - cent[d];
            dist += diff * diff;
        }

        if (dist < best_d) {
            best_d = dist;
            best_k = c;
        }
    }

    labels[idx] = best_k;
}

__global__ void accumulate_clusters(const float* pixels, const int* labels,
                                    float* sums, int* counts, size_t N) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= N) {
        return;
    }
    int clus = labels[idx];
    const float* pix = pixels + (idx * PIXEL_DIM);
    for (int d = 0; d < PIXEL_DIM; d++) {
        atomicAdd(&sums[(clus * PIXEL_DIM) + d], pix[d]);
    }
    atomicAdd(&counts[clus], 1);
}
