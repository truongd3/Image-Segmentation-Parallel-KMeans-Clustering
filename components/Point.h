#ifndef POINT_H
#define POINT_H

#include <cmath>
#include <limits>

using namespace std;

class Point {
private:
    double x, y;    // coordinates
    int cluster;    // no default cluster
    double minDist; // default infinite dist to nearest cluster

public:
    Point() : 
        x(0.0),
        y(0.0), 
        cluster(-1),
        minDist(__DBL_MAX__) {}

    Point(double x, double y) : 
        x(x), 
        y(y),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    double getX() const { return x; }
    double getY() const { return y; }
    int getCluster() const { return cluster; }
    double getMinDist() const { return minDist; }

    void setX(double x) { x = x; }
    void setY(double y) { y = y; }
    void setCluster(int c) { cluster = c; }
    void setMinDist(double d) { minDist = d; }

    double distance(const Point& p) {
        return sqrt((p.x - x) * (p.x - x) + (p.y - y) * (p.y - y));
    }
};

#endif // POINT_H
