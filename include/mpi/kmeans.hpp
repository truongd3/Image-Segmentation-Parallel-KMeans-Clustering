#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

namespace mpi {
inline constexpr int MAX_ITER = 100;
inline constexpr double TOL = 1e-4;

class Kmeans {
public:
    Kmeans(int num_clusters, int max_iterations = MAX_ITER, double tol = TOL);

    void fit(const vector<cv::Vec3f> &points, vector<int> &labels);

    const vector<cv::Vec3f> &getCenters() const;

private:
    int k;
    int max_iters;
    double tol;
    vector<cv::Vec3f> centers;

    static double sqDist(const cv::Vec3f &a, const cv::Vec3f &b);
};
} // namespace mpi
