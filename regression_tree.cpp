#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

// Structure to represent a single data point (Feature, Target)
struct DataPoint {
    double x; // Feature
    double y; // Target value
};

// Structure for the Tree Node
struct Node {
    bool is_leaf;
    double split_value;
    double predicted_value; // Only used if it's a leaf node
    Node* left;
    Node* right;

    Node() : is_leaf(false), split_value(0.0), predicted_value(0.0), left(nullptr), right(nullptr) {}
};

// Helper function to calculate the variance (or MSE) of target values
double calculate_variance(const std::vector<DataPoint>& data) {
    if (data.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& point : data) sum += point.y;
    double mean = sum / data.size();

    double variance = 0.0;
    for (const auto& point : data) {
        variance += (point.y - mean) * (point.y - mean);
    }
    return variance / data.size();
}

// Helper function to calculate the mean of target values
double calculate_mean(const std::vector<DataPoint>& data) {
    if (data.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& point : data) sum += point.y;
    return sum / data.size();
}

// Function to build the regression tree recursively
Node* build_tree(std::vector<DataPoint> data, int max_depth, int current_depth = 0) {
    Node* node = new Node();

    // Base cases: Stop splitting if max depth is reached or variance is 0 (or data is too small)
    if (current_depth >= max_depth || data.size() <= 2 || calculate_variance(data) < 1e-6) {
        node->is_leaf = true;
        node->predicted_value = calculate_mean(data);
        return node;
    }

    // Sort data based on the feature X to easily find the best split point
    std::sort(data.begin(), data.end(), [](const DataPoint& a, const DataPoint& b) {
        return a.x < b.x;
    });

    double best_variance_reduction = -1.0;
    double best_split_value = data[0].x;
    size_t best_split_index = 0;
    double current_variance = calculate_variance(data);

    // Iterate through data points to find the split that minimizes weighted variance of children
    for (size_t i = 1; i < data.size(); ++i) {
        std::vector<DataPoint> left_subset(data.begin(), data.begin() + i);
        std::vector<DataPoint> right_subset(data.begin() + i, data.end());

        double left_var = calculate_variance(left_subset);
        double right_var = calculate_variance(right_subset);

        // Weighted variance of the split
        double weight_left = (double)left_subset.size() / data.size();
        double weight_right = (double)right_subset.size() / data.size();
        double total_split_variance = (weight_left * left_var) + (weight_right * right_var);

        double variance_reduction = current_variance - total_split_variance;

        if (variance_reduction > best_variance_reduction) {
            best_variance_reduction = variance_reduction;
            best_split_index = i;
            // Split value is the midpoint between the two feature values
            best_split_value = (data[i-1].x + data[i].x) / 2.0;
        }
    }

    // If no meaningful split is found, make it a leaf
    if (best_variance_reduction <= 0) {
        node->is_leaf = true;
        node->predicted_value = calculate_mean(data);
        return node;
    }

    // Assign split and recursively build left and right subtrees
    node->split_value = best_split_value;
    std::vector<DataPoint> left_data(data.begin(), data.begin() + best_split_index);
    std::vector<DataPoint> right_data(data.begin() + best_split_index, data.end());

    node->left = build_tree(left_data, max_depth, current_depth + 1);
    node->right = build_tree(right_data, max_depth, current_depth + 1);

    return node;
}

// Function to predict a value using the trained tree
double predict(Node* root, double x) {
    if (root->is_leaf) {
        return root->predicted_value;
    }
    if (x <= root->split_value) {
        return predict(root->left, x);
    } else {
        return predict(root->right, x);
    }
}

// Helper to free memory
void delete_tree(Node* node) {
    if (node == nullptr) return;
    delete_tree(node->left);
    delete_tree(node->right);
    delete node;
}

int main() {
    // Sample Dataset: trying to learn a basic trend (e.g., y ≈ 2x)
    std::vector<DataPoint> dataset = {
        {1.0, 2.2}, {2.0, 3.8}, {3.0, 6.5}, {4.0, 7.9}, {5.0, 10.1}
    };

    // Build a tree with a maximum depth of 3
    Node* root = build_tree(dataset, 3);

    // Test predictions
    std::vector<double> test_inputs = {1.5, 3.2, 4.8};

    std::cout << "--- Regression Tree Predictions ---" << std::endl;
    for (double test_x : test_inputs) {
        double prediction = predict(root, test_x);
        std::cout << "Input X: " << test_x << " -> Predicted Y: " << prediction << std::endl;
    }

    // Clean up memory
    delete_tree(root);
    return 0;
}