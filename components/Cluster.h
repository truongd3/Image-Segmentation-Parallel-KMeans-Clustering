#include <vector>
#include "Point.h"

using namespace std;

class Cluster {
private:
    int clusterId;
    vector<double> centroid;
    vector<Point> points;

public:
    Cluster(int clusterId, Point centroid) {
        // this->clusterId = clusterId;
        // for (int i = 0; i < centroid.getDimensions(); i++) {
        //     this->centroid.push_back(centroid.getVal(i));
        // }
        // this->addPoint(centroid);
    }
};