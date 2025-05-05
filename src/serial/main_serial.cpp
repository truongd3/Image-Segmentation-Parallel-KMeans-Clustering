#include "serial/img_seg_serial.hpp"
#include <iostream>
#include <ctime>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <image_path> <k> <output_path>\n";
        return 1;
    }

    string image_path = argv[1];
    int k = stoi(argv[2]);
    string output_path = argv[3];

    clock_t start_time = clock();
    bool success = serial::imgSeg(image_path, k, output_path);
    clock_t end_time = clock();

    if (success) {
        cout << "Image segmentation completed successfully. Output saved to " << output_path << '\n';
        double secs = double(end_time - start_time) / CLOCKS_PER_SEC;
        cout << "Serial version -> Total segmentation time: " << secs << " seconds\n";
    } else {
        cerr << "Image segmentation failed.\n";
        return 1;
    }
    return 0;
}
