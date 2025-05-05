#include "common/constants.hpp"
#include "common/image_utils.hpp"
#include "common/kmeans_utils.hpp"
#include "mpi_gpu/kmeans_mpi_gpu.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank{};
    int nprocs{};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <image_path> <k> <output_path>\n";
        MPI_Finalize();
        return 1;
    }

    string image_path = argv[1];
    size_t K = stoi(argv[2]);
    string output_path = argv[3];

    vector<float> h_pixels;
    int rows{};
    int cols{};
    int N{};

    // rank 0 load image and flatten that image into vector<float>
    if (rank == 0) {
        cv::Mat img = image_utils::load_float_image(image_path);
        rows = img.rows;
        cols = img.cols;
        N = rows * cols;
        h_pixels = image_utils::flatten_image(img, N);
    }

    // broadcast image dimension info to other ranks
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    N = rows * cols;

    // root init centroids and then broadcast to other ranks
    vector<float> h_centroids(K * PIXEL_DIM);
    if (rank == 0) kmeans_utils::init_centroids(h_pixels, h_centroids, N, K);
    MPI_Bcast(h_centroids.data(), PIXEL_DIM * K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter h_pixels to all ranks, each rank work is stored in h_local_pixels
    int base = N / nprocs;
    int rem = N % nprocs;
    int my_work = base + (rank < rem ? 1 : 0);

    vector<int> elems_count(nprocs);
    vector<int> displs(nprocs);
    vector<int> lb_count(nprocs);
    vector<int> lb_displs(nprocs);
    for (int i = 0, offset = 0; i < nprocs; i++) {
        int cnt = base + (i < rem ? 1 : 0);
        elems_count[i] = cnt * 3;
        displs[i] = offset * 3;
        lb_count[i] = cnt;
        lb_displs[i] = offset;
        offset += cnt;
    }
    vector<float> h_local_pixels(elems_count[rank]);
    MPI_Scatterv(h_pixels.data(), elems_count.data(), displs.data(), MPI_FLOAT,
                 h_local_pixels.data(), elems_count[rank], MPI_FLOAT, 0,
                 MPI_COMM_WORLD);

    // kmeans routine
    vector<int> all_labels(N);

    // synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    kmeans_mpi_gpu(K, my_work, h_local_pixels, h_centroids, all_labels, rank, lb_count, lb_displs);
    if (rank == 0) image_utils::produce_image(rows, cols, all_labels, h_centroids, output_path);
    // wait for everyone to finish
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    cout << "Hybrid version -> Rank #" << rank << " segmentation time: " << (end_time - start_time) << " seconds\n";

    MPI_Finalize();

    return 0;
}
