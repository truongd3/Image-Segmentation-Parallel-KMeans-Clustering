#include "mpi/img_seg_mpi.hpp"
#include <iostream>
#include <mpi.h>
#include <ctime>

using namespace std;

int main(int argc, char **argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <image_path> <k> <output_path>" << '\n';
        return 1;
    }

    MPI_Init(&argc, &argv);

    string image_path = argv[1], output_path = argv[3];
    int k = stoi(argv[2]);

    clock_t start_time = clock();
    bool success = mpi::imgSeg(image_path, k, output_path);
    clock_t end_time = clock();

    if (success) {
        cout << "Image segmentation completed successfully. Output saved to " << output_path << '\n';
        double secs = double(end_time - start_time) / CLOCKS_PER_SEC;
        cout << "MPI version -> Total segmentation time: " << secs << " seconds\n";
    } else {
        cerr << "Image segmentation failed." << '\n';
        return 1;
    }

    MPI_Finalize();
    return 0;
}
