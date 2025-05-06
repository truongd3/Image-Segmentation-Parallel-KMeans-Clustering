#include "mpi/kmeans.hpp"
#include <mpi.h>
#include <random>

using namespace std;

namespace mpi {

Kmeans::Kmeans(int num_clusters, int max_iterations, double tol)
    : k(num_clusters), max_iters(max_iterations), tol(tol) {}

inline double Kmeans::sqDist(const cv::Vec3f& a, const cv::Vec3f& b) {
    cv::Vec3f diff = a - b;
    return diff.dot(diff);
}

void Kmeans::fit(const vector<cv::Vec3f>& points, vector<int>& labels) {
    int rank = 0, num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const int N = (int)(points.size());
    labels.assign(N, -1);

    // Initialize centers on root process
    centers = vector<cv::Vec3f>(k);
    if (rank == 0) {
        mt19937 rng(12345);
        uniform_int_distribution<int> dist(0, N - 1);
        for (int j = 0; j < k; ++j) centers[j] = points[dist(rng)];
    }

    // Broadcast initial centers
    MPI_Bcast(centers.data(), k * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    vector<cv::Vec3f> new_centers(k);
    vector<cv::Vec3f> local_sums(k);
    vector<cv::Vec3f> global_sums(k);
    vector<int> local_counts(k);
    vector<int> global_counts(k);

    for (int iter = 0; iter < max_iters; iter++) {
        for (int j = 0; j < k; ++j) { // Reset accumulators
            local_sums[j] = cv::Vec3f(0, 0, 0);
            local_counts[j] = 0;
        }

        // Assignment step: each process handles a strided portion
        for (int i = rank; i < N; i += num_procs) {
            int best_cluster = 0;
            double best_distance = sqDist(points[i], centers[0]);
            for (int j = 1; j < k; ++j) {
                double d = sqDist(points[i], centers[j]);
                if (d < best_distance) {
                    best_distance = d;
                    best_cluster = j;
                }
            }
            labels[i] = best_cluster;
            local_counts[best_cluster]++;
            local_sums[best_cluster] += points[i];
        }

        // Reduce sums and counts across all processes
        MPI_Allreduce(local_sums.data(), global_sums.data(), k * 3, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Compute new centers and track maximum shift
        // double max_shift2 = 0.0;
        for (int j = 0; j < k; ++j) {
            if (global_counts[j] > 0) new_centers[j] = global_sums[j] * (1.0f / global_counts[j]);
            else new_centers[j] = centers[j];

            // double shift = sqDist(centers[j], new_centers[j]);
            // if (shift > max_shift2) max_shift2 = shift;
        }

        // Update centers
        centers = new_centers;
        MPI_Bcast(centers.data(), k * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // if (sqrt(max_shift2) < tol) break;
    }

    // Gather final labels: each process has its subset in labels, others are -1
    vector<int> global_labels(N, -1);
    MPI_Allreduce(labels.data(), global_labels.data(), N, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    labels = move(global_labels);
}

const vector<cv::Vec3f>& Kmeans::getCenters() const { return centers; }

} // namespace mpi
