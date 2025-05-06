#include "serial/kmeans.hpp"
#include <limits>

namespace serial {
Kmeans::Kmeans(int num_clusters, int max_iterations, double tol)
    : k(num_clusters), max_iters(max_iterations), tol(tol) {}

double Kmeans::sqDist(const cv::Vec3f& a, const cv::Vec3f& b) {
    cv::Vec3f diff = a - b;
    return diff.dot(diff);
}

void Kmeans::fit(const std::vector<cv::Vec3f>& points,
                 std::vector<int>& labels) {
    size_t n_points = points.size();
    labels.resize(n_points, 0);

    centers.clear();
    for (int i = 0; i < k; i++) {
        int index = rand() % (int)n_points;
        centers.push_back(points[index]);
    }

    for (int iter = 0; iter < max_iters; iter++) {
        std::cout << "Iteration: " << iter << "\n";
        bool change = false;
        for (size_t i = 0; i < n_points; i++) {
            int best_cluster = 0;
            double best_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < k; j++) {
                double dist = sqDist(points[i], centers[j]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_cluster = j;
                }
            }

            if (labels[i] != best_cluster) {
                labels[i] = best_cluster;
                change = true;
            }
        }

        // Update
        std::vector<cv::Vec3f> new_centers(k, cv::Vec3f(0, 0, 0));
        std::vector<int> counts(k, 0);

        for (size_t i = 0; i < n_points; i++) {
            int cur_cluster = labels[i];
            new_centers[cur_cluster] += points[i];
            counts[cur_cluster] += 1;
        }

        for (int j = 0; j < k; j++) {
            if (counts[j] > 0) {
                new_centers[j] *= (1.0 / counts[j]);
            } else {
                new_centers[j] = points[rand() % n_points];
            }
        }

        // double center_shift = 0.0;
        // for (int j = 0; j < k; ++j) {
        //     center_shift += sqDist(centers[j], new_centers[j]);
        // }

        centers = new_centers;

        // if (center_shift < tol || !change) {
        //     break;
        // }
    }
}

const std::vector<cv::Vec3f>& Kmeans::getCenters() const { return centers; }

} // namespace serial
