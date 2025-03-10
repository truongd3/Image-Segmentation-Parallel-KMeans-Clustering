#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "components/Point.h"

using namespace std;

vector<Point> readcsv(string filePath) {
    vector<Point> points;
    string line;
    ifstream file(filePath);

    while (getline(file, line)) {
        stringstream lineStream(line);
        string bit;
        double x, y;
        getline(lineStream, bit, ',');
        x = stof(bit);
        getline(lineStream, bit, '\n');
        y = stof(bit);

        points.push_back(Point(x, y));
    }
    return points;
}

/**
 * Perform k-means clustering
 * @param points - pointer to vector of points
 * @param epochs - number of k means iterations
 * @param k - the number of initial centroids
 */
void kMeansClustering(vector<Point>* points, int epochs, int k) {
    int n = points->size();

    // Randomly initialise centroids
    // The index of the centroid within the centroids vector represents the cluster label.
    vector<Point> centroids;
    srand(time(0));
    for (int i = 0; i < k; i++) centroids.push_back(points->at(rand() % n));

    for (int i = 0; i < epochs; ++i) {
        // For each centroid, compute distance from centroid to each point and update point's cluster if necessary
        for (vector<Point>::iterator c = begin(centroids); c != end(centroids); c++) {
            int clusterId = c - begin(centroids);

            for (vector<Point>::iterator it = points->begin(); it != points->end(); it++) {
                Point p = *it;
                double dist = c->distance(p);
                if (dist < p.getMinDist()) {
                    p.setMinDist(dist);
                    p.setCluster(clusterId);
                }
                *it = p;
            }
        }

        // Create vectors to keep track of data needed to compute means
        vector<int> nPoints;
        vector<double> sumX, sumY;
        for (int j = 0; j < k; ++j) {
            nPoints.push_back(0);
            sumX.push_back(0.0);
            sumY.push_back(0.0);
        }

        // Iterate over points to append data to centroids
        for (vector<Point>::iterator it = points->begin(); it != points->end(); it++) {
            int clusterId = (*it).getCluster();
            nPoints[clusterId] += 1;
            sumX[clusterId] += (*it).getX();
            sumY[clusterId] += (*it).getY();

            (*it).setMinDist(__DBL_MAX__);  // reset distance
        }
        // Compute the new centroids
        for (vector<Point>::iterator c = begin(centroids); c != end(centroids); c++) {
            int clusterId = c - begin(centroids);
            (*c).setX(sumX[clusterId] / nPoints[clusterId]);
            (*c).setY(sumY[clusterId] / nPoints[clusterId]);
        }
    }

    // Write to csv
    ofstream myfile;
    myfile.open("output.csv");
    myfile << "x,y,c" << endl;

    for (vector<Point>::iterator it = points->begin(); it != points->end(); it++) {
        myfile << (*it).getX() << "," << (*it).getY() << "," << (*it).getCluster() << endl;
    }
    myfile.close();
}

int main() {
    vector<Point> points = readcsv("db/points.csv");
    kMeansClustering(&points, 100, 5);
}