#include <vector>
#include <algorithm>
#include "Point.h"

using namespace std;

class Cluster {
private:
    int clusterId;
    Point center;
    vector<Point> points;

public:
    Cluster() :
        clusterId(-1),
        center() {}

    Cluster(int clusterId, const Point& centroid) : // Point thôi?
        clusterId(clusterId),
        center(centroid) {}

    const Point& getCenter() const { return center; }
    const vector<Point>& getPoints() const { return points; }

    void setCenter(const Point& new_center) { center = new_center; }

    void addPoint(const Point& point) {
        points.push_back(point);
    }

    void clearPoints() {
        points.clear();
    }
};