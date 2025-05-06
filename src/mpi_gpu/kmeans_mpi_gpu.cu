#include "common/constants.hpp"
#include "gpu/kmeans_kernels.cuh"
#include <iostream>
#include <mpi.h>
#include <vector>

using namespace std;

#define CHECK_CUDA(err)                                                        \
    if (err != cudaSuccess) {                                                  \
        std::cerr << "CUDA error " << cudaGetErrorString(err) << " at "        \
                  << __LINE__ << '\n';                                         \
        exit(1);                                                               \
    }

__constant__ float constCentroids[MAX_K * PIXEL_DIM];

void kmeans_mpi_gpu(size_t K, size_t N, const vector<float>& h_pixels,
                    vector<float>& h_centroids,
                    vector<int>& all_labels, int my_rank,
                    const vector<int>& lb_count,
                    const vector<int>& lb_displs) {
    float* d_pixels{};
    float* d_sums{};
    int* d_labels{};
    int* d_counts{};

    const size_t d_pixels_size = N * PIXEL_DIM * sizeof(float);
    const size_t d_centroids_size = K * PIXEL_DIM * sizeof(float);
    const size_t d_labels_size = N * sizeof(int);
    const size_t d_sums_size = K * PIXEL_DIM * sizeof(float);
    const size_t d_counts_size = K * sizeof(int);

    // Allocate global mem for input arrays
    CHECK_CUDA(cudaMalloc(&d_pixels, d_pixels_size));
    CHECK_CUDA(cudaMalloc(&d_labels, d_labels_size));
    CHECK_CUDA(cudaMalloc(&d_sums, d_sums_size));
    CHECK_CUDA(cudaMalloc(&d_counts, d_counts_size));

    CHECK_CUDA(cudaMemcpy(d_pixels, h_pixels.data(), d_pixels_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(constCentroids, h_centroids.data(), d_centroids_size));

    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    vector<float> h_sums(K * PIXEL_DIM);
    vector<int> h_counts(K);
    vector<int> h_labels(N);

    size_t sharedBytes = (MAX_K * PIXEL_DIM) * sizeof(float) + MAX_K * sizeof(int);

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        // reset sums and counts for new interation
        CHECK_CUDA(cudaMemset(d_sums, 0, d_sums_size));
        CHECK_CUDA(cudaMemset(d_counts, 0, d_counts_size));

        assign_and_reduce<MAX_K><<<gridDim, blockDim, sharedBytes>>>(d_pixels, d_labels, d_sums, d_counts, N, K);

        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_sums.data(), d_sums, d_sums_size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_counts.data(), d_counts, d_counts_size, cudaMemcpyDeviceToHost));

        // compute global sums and counts
        MPI_Allreduce(MPI_IN_PLACE, h_sums.data(), PIXEL_DIM * K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, h_counts.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // root find new centroids and broadcast to all ranks
        // bool converged = true;
        if (my_rank == 0) {
            for (int clus = 0; clus < K; ++clus) {
                if (h_counts[clus] == 0)
                    continue;
                for (int d = 0; d < PIXEL_DIM; d++) {
                    float new_clus_comp = h_sums[(clus * PIXEL_DIM) + d] / (float)h_counts[clus];
                    // if (abs(new_clus_comp - h_centroids[(clus * PIXEL_DIM) +
                    // d]) > TOL) converged = false;
                    h_centroids[(clus * PIXEL_DIM) + d] = new_clus_comp;
                }
            }
        }

        // MPI_Bcast(&converged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        MPI_Bcast(h_centroids.data(), PIXEL_DIM * K, MPI_FLOAT, 0, MPI_COMM_WORLD);
        CHECK_CUDA(cudaMemcpyToSymbol(constCentroids, h_centroids.data(),
                                      d_centroids_size));
        // if (converged) {
        //     cout << "Converged at iteration " << iter << "\n";
        //     break;
        // }
    }

    CHECK_CUDA(cudaMemcpy(h_labels.data(), d_labels, d_labels_size, cudaMemcpyDeviceToHost));

    MPI_Gatherv(h_labels.data(), N, MPI_INT, all_labels.data(), lb_count.data(), lb_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    cudaFree(d_pixels);
    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);
}
