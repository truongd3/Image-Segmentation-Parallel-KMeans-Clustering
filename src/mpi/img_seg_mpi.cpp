#include "common/image_utils.hpp"
#include "mpi/img_seg_mpi.hpp"
#include "mpi/kmeans.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

namespace mpi {
bool imgSeg(const string &image_path, int k, const string &output_path) {
    cv::Mat img_float;
    int num_rows{0};

    vector<cv::Vec3f> pixels = image_utils::loadImage(image_path, img_float, num_rows);

    vector<int> labels;
    mpi::Kmeans kmeans(k);
    kmeans.fit(pixels, labels);
    vector<cv::Vec3f> centers = kmeans.getCenters();

    for (int i = 0; i < img_float.rows; i++) {
        int cluster_idx = labels[i];
        img_float.at<cv::Vec3f>(i, 0) = centers[cluster_idx];
    }

    cv::Mat segmented = img_float.reshape(3, num_rows);
    segmented.convertTo(segmented, CV_8U);

    if (!cv::imwrite(output_path, segmented)) {
        cerr << "Error: Could not save the segmented image to " << output_path << '\n';
        return false;
    }
    return true;
}
} // namespace mpi
