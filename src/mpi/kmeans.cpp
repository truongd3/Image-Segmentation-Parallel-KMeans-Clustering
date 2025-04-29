#include "mpi/kmeans.hpp"
#include <limits>
#include <mpi.h>

using namespace std;

namespace mpi {
Kmeans::Kmeans(int num_clusters, int max_iterations, double tol)
    : k(num_clusters), max_iters(max_iterations), tol(tol) {}

double Kmeans::sqDist(const cv::Vec3f &a, const cv::Vec3f &b) {
    cv::Vec3f diff = a - b;
    return diff.dot(diff);
}

void Kmeans::fit(const vector<cv::Vec3f> &points, vector<int> &labels) {
    int n_points = points.size();
    labels.resize(n_points, 0);

    centers.clear();
    // Initialize centers
    for (int i = 0; i < k; i++) {
        int index = rand() % (int)n_points;
        centers.push_back(points[index]);
    }

    for (int iter = 0; iter < max_iters; iter++) {
        int my_rank, num_procs = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        int my_work = n_points / num_procs; // chunk's size
        int start = my_rank * my_work;
        int end = (my_rank == num_procs - 1) ? n_points : start + my_work;

        vector<int> local_labels(my_work, 0);
        bool local_change = false;

        MPI_Bcast(centers, n_points, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = start; i < end; i++) {
            int best_cluster = 0;
            double min_dist = numeric_limits<double>::max();
            for (int j = 0; j < k; j++) {
                double dist = sqDist(points[i], centers[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            int local_index = i - start;
            if (labels[i] != best_cluster) {
                // labels[i] = best_cluster;
                local_labels[local_index] = best_cluster;
                local_change = true;
            } else {
                local_labels[local_index] = labels[i];
            }
        }

        MPI_Gather(local_labels.data(), my_work, MPI_INT, labels.data(), my_work, MPI_INT, 0, MPI_COMM_WORLD);

        int global_change = 0, local_change_int = (int)local_change;
        bool change = global_change != 0;

        // Update
        vector<cv::Vec3f> new_centers(k, cv::Vec3f(0, 0, 0));
        vector<int> counts(k, 0);

        for (int i = 0; i < n_points; i++) {
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

        double center_shift = 0.0;
        for (int j = 0; j < k; ++j) {
            center_shift += sqDist(centers[j], new_centers[j]);
        }

        centers = new_centers;

        if (center_shift < tol || !change) {
            break;
        }
    }
}

const vector<cv::Vec3f> &Kmeans::getCenters() const { return centers; }

} // namespace mpi
