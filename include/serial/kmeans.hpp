#pragma once
#include <opencv2/opencv.hpp>
#include <vector>


namespace serial {
inline constexpr int MAX_ITER = 100;
inline constexpr double TOL = 1e-4;

class Kmeans {
public:
    Kmeans(int num_clusters, int max_iterations = MAX_ITER, double tol = TOL);

    void fit(const std::vector<cv::Vec3f>& points, std::vector<int>& labels);

    const std::vector<cv::Vec3f>& getCenters() const;

private:
    int k;
    int max_iters;
    double tol;
    std::vector<cv::Vec3f> centers;

    static double sqDist(const cv::Vec3f& a, const cv::Vec3f& b);
};
} // namespace serial
