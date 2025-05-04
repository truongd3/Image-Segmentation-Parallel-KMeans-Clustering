#pragma once
#include <cstddef>
#include <vector>

namespace gpu {
void img_seg_gpu(size_t K, size_t N, const std::vector<float>& h_pixels,
                 std::vector<float>& h_centroids, std::vector<int>& h_labels);

void kmeans_mpi_gpu(size_t K, size_t N, const std::vector<float>& h_pixels,
                    std::vector<float>& h_centroids,
                    std::vector<int>& all_labels, int my_rank,
                    const std::vector<int>& elems_count,
                    const std::vector<int>& displs);
} // namespace gpu
