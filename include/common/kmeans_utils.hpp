#pragma once

#include <cstddef>
#include <vector>

namespace kmeans_utils {

void init_centroids(const std::vector<float>& h_pixels,
                    std::vector<float>& h_centroids, size_t N, size_t K);

} // namespace kmeans_utils
