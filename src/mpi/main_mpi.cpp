#include "mpi/img_seg_mpi.hpp"
#include <ctime>
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <k> <output_path>"
                  << '\n';
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank{};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string image_path = argv[1];
    std::string output_path = argv[3];
    int k = std::stoi(argv[2]);

    // synchronize all ranks before we start timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    // clock_t start_time = clock();
    bool success = mpi::imgSeg(image_path, k, output_path);
    // clock_t end_time = clock();
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (success) {
        std::cout << "Image segmentation completed successfully. Output saved to "
             << output_path << '\n';
        double secs = end_time - start_time;
        std::cout << "MPI version -> Rank #" << rank
             << " segmentation time: " << secs << " seconds\n";
    } else {
        std::cerr << "Image segmentation failed." << '\n';
        return 1;
    }

    MPI_Finalize();
    return 0;
}
