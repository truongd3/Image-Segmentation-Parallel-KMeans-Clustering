#include <vector>
#include <algorithm>
#include "components/Cluster.h"

using namespace std;

void assignClusters(vector<Point>& points, vector<Cluster>& clusters) {
    for (auto& point : points) {
        vector<double> dist;
        for (const auto& cluster : clusters) {
            dist.push_back(point.distance(cluster.getCenter()));
        }

        int curr_cluster = min_element(dist.begin(), dist.end()) - dist.begin();
        clusters[curr_cluster].addPoint(point); // add Point to Cluster
        point.setCluster(curr_cluster); // assign clusterId to Point
    }
}

void updateClusters(vector<Cluster>& clusters) {
    for (auto& cluster : clusters) {
        vector<Point> clusterPoints = cluster.getPoints(); // const &

        if (!clusterPoints.empty()) {
            Point new_center = Point::mean(clusterPoints); 
            cluster.setCenter(new_center); // New cluster center
            cluster.clearPoints(); // Clear points
        }
    }
}

vector<int> predictClusters(const vector<Point>& points, const vector<Cluster>& clusters) {
    vector<int> predictions;

    for (const auto& point : points) {
        vector<double> dist;

        // Calculate the distance from the point to each cluster center
        for (const auto& cluster : clusters) {
            dist.push_back(point.distance(cluster.getCenter()));
        }

        // Find the index of the closest cluster (size_t)
        int closestCluster = min_element(dist.begin(), dist.end()) - dist.begin();
        predictions.push_back(closestCluster);
    }

    return predictions;
}