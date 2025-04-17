#pragma once
#include <string>

namespace serial {

bool imgSeg(const std::string &image_path, int k,
            const std::string &output_path);
}
