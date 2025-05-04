#pragma once
#include <cstddef>
#include <vector>

void kmeans_mpi_gpu(size_t K, size_t N, const std::vector<float>& h_pixels,
                    std::vector<float>& h_centroids,
                    std::vector<int>& all_labels, int my_rank,
                    const std::vector<int>& lb_count,
                    const std::vector<int>& lb_displs);
