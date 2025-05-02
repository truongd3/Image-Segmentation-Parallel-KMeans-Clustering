#include "common/kmeans_utils.hpp"
#include "common/constants.hpp"
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

namespace kmeans_utils {
void init_centroids(const std::vector<float>& h_pixels,
                    std::vector<float>& h_centroids, size_t N, size_t K) {
    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> pick(0, N - 1);

    std::unordered_set<std::size_t> seen;
    while (seen.size() < static_cast<std::size_t>(K)) {
        std::size_t idx = pick(rng);
        if (!seen.insert(idx).second) {
            continue;
        }

        std::size_t c = seen.size() - 1;
        std::copy_n(&h_pixels[idx * PIXEL_DIM], PIXEL_DIM,
                    &h_centroids[c * PIXEL_DIM]);
    }
}
} // namespace kmeans_utils
