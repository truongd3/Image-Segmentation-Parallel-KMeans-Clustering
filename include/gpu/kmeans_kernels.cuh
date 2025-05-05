#pragma once
#include "common/constants.hpp"
#include <cfloat>

__global__ void assign_clusters(int* labels, const float* pixels,
                                const float* centroids, size_t N, size_t K);

__global__ void accumulate_clusters(const float* pixels, const int* labels,
                                    float* sums, int* counts, size_t N);

template<int MAX_K>
__global__ void assign_and_reduce(const float* __restrict__ pixels,
                                  const float* __restrict__ centroids,
                                  int* labels, float* sums, int* counts,
                                  size_t N, size_t K) {
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    size_t tpb = blockDim.x;
    size_t idx = (bid * tpb) + tid;

    if (idx >= N) {
        return;
    }

    extern __shared__ char sdata[];
    float* ssum = reinterpret_cast<float*>(sdata);
    int* scnt = reinterpret_cast<int*>(ssum + (PIXEL_DIM * MAX_K));

    int totalSumElems = static_cast<int>(PIXEL_DIM * K);
    for (size_t i = tid; i < totalSumElems; i += tpb) {
        ssum[i] = 0.0F;
    }
    for (size_t i = tid; i < K; i += tpb) {
        scnt[i] = 0;
    }
    __syncthreads();

    // Label Assignment
    const float* px = pixels + (idx * PIXEL_DIM);
    float best_d = FLT_MAX;
    int best_k = 0;
    for (int c = 0; c < K; ++c) {
        const float* centroid = centroids + (c * PIXEL_DIM);
        float dist = 0.0F;
        for (int d = 0; d < PIXEL_DIM; ++d) {
            float diff = px[d] - centroid[d];
            dist += diff * diff;
        }
        if (dist < best_d) {
            best_d = dist;
            best_k = c;
        }
    }
    labels[idx] = best_k;

    // Accumulate pixel into shared sums and counts
    int startOffset = best_k * PIXEL_DIM;
    for (size_t d = 0; d < PIXEL_DIM; ++d) {
        atomicAdd(&ssum[startOffset + d], px[d]);
    }
    atomicAdd(&scnt[best_k], 1);
    __syncthreads();

    // One thread per block writes shared accumulators to global mem
    if (tid == 0) {
        for (size_t c = 0; c < K; ++c) {
            size_t start = c * PIXEL_DIM;
            atomicAdd(&counts[c], scnt[c]);
            for (size_t d = 0; d < PIXEL_DIM; ++d) {
                atomicAdd(&sums[start + d], ssum[start + d]);
            }
        }
    }
}
