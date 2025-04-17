#include "serial/img_seg_serial.hpp"
#include <iostream>

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <k> <output_path>" << '\n';
        return 1;
    }
    std::string image_path = argv[1];
    int k = std::stoi(argv[2]);
    std::string output_path = argv[3];

    if (serial::imgSeg(image_path, k, output_path)) {
        std::cout
            << "Image segmentation completed successfully. Output saved to "
            << output_path << '\n';
    } else {
        std::cerr << "Image segmentation failed." << '\n';
        return 1;
    }
    return 0;
}
