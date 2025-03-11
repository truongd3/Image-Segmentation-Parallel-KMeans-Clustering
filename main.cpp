#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "kmeans.h"

using namespace std;

vector<Point> readcsv(const string& filePath) {
    vector<Point> points;
    ifstream file(filePath);

    string line;
    while (getline(file, line)) {
        stringstream lineStream(line);
        string bit;

        double x, y;
        if (getline(lineStream, bit, ',')) x = stof(bit);
        if (getline(lineStream, bit, '\n')) y = stof(bit);

        points.push_back(Point(x, y));
    }
    return points;
}

int epochs = 0; // Number of epochs
int k = 0;      // Number of clusters

int main(int argc, char *argv[]) { // ./main <epochs> <k>
    srand(23);
    
    if (argc == 3) {
        epochs = stoi(argv[1]);
        k = stoi(argv[2]);
    } else {
        cerr << "Usage: ./main <epochs> <k>" << endl;
        return 1;
    }

    vector<Point> points = readcsv("db/input.csv");
    int n = points.size();

    if (n == 0 || k <= 0) {
        cerr << "Invalid dataset or cluster count." << endl;
        return 1;
    }

    vector<Cluster> clusters;
    // Initialize k random points as centers of clusters
    for (int i = 0; i < k; ++i) {
        clusters.push_back(Cluster(i, points[rand() % n]));
    }

    // Run K-Means algorithm
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (auto &cluster : clusters) {
            cluster.clearPoints();
        }

        assignClusters(points, clusters);
        updateClusters(clusters);
    }

    cout << "Final cluster centers:\n";
    for (const auto& cluster : clusters) {
        cout << "Cluster " << cluster.getCenter().getX() << ", " << cluster.getCenter().getY() << endl;
    }

    // Predict clusters for the input points
    vector<int> predictions = predictClusters(points, clusters);
    cout << "\nPoint assignments:\n";
    for (size_t i = 0; i < points.size(); ++i) {
        cout << "Point (" << points[i].getX() << ", " << points[i].getY() << ") -> Cluster " << predictions[i] << endl;
    }

    return 0;
}
