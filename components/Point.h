#ifndef POINT_H
#define POINT_H

#include <cmath>
#include <limits>
#include <vector>

using namespace std;

class Point {
private:
    double x, y, z; // 3D coordinates
    int cluster;    // no default cluster
    double minDist; // default infinite dist to nearest cluster

public:
    Point() : 
        x(0.0), y(0.0), z(0.0),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    Point(double x, double y, double z) : 
        x(x), y(y), z(z),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }
    int getCluster() const { return cluster; }
    double getMinDist() const { return minDist; }

    void setX(double x) { this->x = x; }
    void setY(double y) { this->y = y; }
    void setZ(double z) { this->z = z; }
    void setCluster(int c) { cluster = c; }
    void setMinDist(double d) { minDist = d; }

    double distance(const Point& p) const {
        return sqrt((p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z));
    }

    static Point mean(const vector<Point>& points) {
        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        for (const auto& point : points) {
            sum_x += point.x;
            sum_y += point.y;
            sum_z += point.z;
        }
        int count = points.size();
        return Point(sum_x / count, sum_y / count, sum_z / count);
    }
};

#endif // POINT_H
