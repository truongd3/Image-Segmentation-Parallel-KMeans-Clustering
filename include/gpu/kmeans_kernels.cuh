#pragma once
__global__ void assign_clusters(int* labels, const float* pixels,
                                const float* centroids, size_t N, size_t K);

__global__ void accumulate_clusters(const float* pixels, const int* labels,
                                    float* sums, int* counts, size_t N);
