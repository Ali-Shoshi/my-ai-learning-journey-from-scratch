#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <map>

/*
 * Machine Learning Profile:
 * ----------------------------------------------------------------------
 * - Model: k-Nearest Neighbors (k-NN) (Instance-Based / Non-Parametric Classifier)
 * - Optimizer: None (Lazy Learner / Memorizes training instances)
 * - Batch Method: Lazy Evaluation (Calculates distances dynamically during inference)
 * - Regularization: Hyperparameter k (Higher k values smooth out the decision boundaries)
 * - Distance Math: Standard L2 Euclidean Metric: sqrt((x1 - x2)^2 + (y1 - y2)^2)
 * - Metric: Plurality Majority Voting among the k-closest localized coordinates
 * ----------------------------------------------------------------------
 * This program implements a k-Nearest Neighbors (k-NN) classification algorithm from scratch 
 * to handle multi-class spatial recognition. It automatically builds a synthetic dataset 
 * consisting of 120 geometric coordinate records grouped into 3 distinct overlapping circular 
 * clusters centered symmetrically in a 2D vector space. Unlike parametric models, this structure 
 * requires no active training epoch phases; instead, it saves the reference space directly to 
 * memory. During the query prediction runtime, the classifier computes the explicit Euclidean 
 * distance from an unknown point to every instance in the dataset, isolates the top k (k=5) 
 * closest neighbors via structural sorting, and classifies the target using a majority vote.
 */

const double PI = 3.14159265358979323846;

struct Point {
    double x, y; 
    int label;  
};

struct DistanceResult {
    double distance;
    int label;
    bool operator<(const DistanceResult& other) const {
        return distance < other.distance;
    }
};

double calculateDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

int knnClassify(const std::vector<Point>& dataset, const Point& query, int k) {
    std::vector<DistanceResult> distances;

    for (const auto& point : dataset) {
        double dist = calculateDistance(query, point);
        distances.push_back({dist, point.label});
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> votes(3, 0);

    for (int i = 0; i < k; ++i) {
        votes[distances[i].label]++;
    }

    auto maxIt = std::max_element(votes.begin(), votes.end());

    return std::distance(votes.begin(), maxIt);
}

std::vector<Point> generateThreeCirclesData(int pointsPerClass) {
    std::vector<Point> dataset;
    std::mt19937 gen(1337); 
    std::uniform_real_distribution<> dis_angle(0.0, 2.0 * PI);
    
    std::uniform_real_distribution<> dis_radius(0.0, 1.0); 

    double centers[3][2] = {
        {0.0, 0.0},          
        {3.0, 0.0},          
        {1.5, 2.598}         
    };

    for (int label = 0; label < 3; ++label) {
        double cx = centers[label][0];
        double cy = centers[label][1];

        for (int i = 0; i < pointsPerClass; ++i) {
            double angle = dis_angle(gen);
            double r = dis_radius(gen); 

            double x = cx + r * std::cos(angle);
            double y = cy + r * std::sin(angle);

            dataset.push_back({x, y, label});
        }
    }

    return dataset;
}

int main() {
    system("cls");
    std::vector<Point> dataset = generateThreeCirclesData(40);

    int k = 5;

    std::vector<Point> queries = {
        {0.2, 0.1, -1},   
        {2.9, -0.2, -1},  
        {1.4, 2.3, -1},   
        {1.5, 1.0, -1}    
    };

    std::cout << "--- 3-Circle k-NN Classification Results (k = " << k << ") ---\n\n";

    for (size_t i = 0; i < queries.size(); ++i) {
        int predictedLabel = knnClassify(dataset, queries[i], k);
        std::cout << "Query " << i + 1 << " (" << queries[i].x << ", " << queries[i].y << ") "
                  << "-> Predicted Class: " << predictedLabel << "\n";
    }

    return 0;
}