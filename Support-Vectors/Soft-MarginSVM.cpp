#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_POINTS 100
#define EPOCHS 1000
#define LEARNING_RATE 0.01

typedef struct {
    double x[2];
    int label; // +1 or -1
} Point;

// Train the SVM model using Gradient Descent
void train_svm(Point data[], double c_param, double *out_w, double *out_b) {
    double w[2] = {0.0, 0.0};
    double b = 0.0;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < NUM_POINTS; i++) {
            double decision = data[i].label * (w[0] * data[i].x[0] + w[1] * data[i].x[1] + b);

            if (decision < 1.0) {
                w[0] = w[0] - LEARNING_RATE * (w[0] / c_param - data[i].label * data[i].x[0]);
                w[1] = w[1] - LEARNING_RATE * (w[1] / c_param - data[i].label * data[i].x[1]);
                b = b + LEARNING_RATE * data[i].label * c_param;
            } else {
                w[0] = w[0] - LEARNING_RATE * (w[0] / c_param);
                w[1] = w[1] - LEARNING_RATE * (w[1] / c_param);
            }
        }
    }

    out_w[0] = w[0];
    out_w[1] = w[1];
    *out_b = b;
}

// Evaluation Function
void evaluate_model(Point data[], double w[2], double b, const char* model_name) {
    int correct_predictions = 0;
    int margin_violations = 0;

    for (int i = 0; i < NUM_POINTS; i++) {
        // Compute raw score: w . x + b
        double score = w[0] * data[i].x[0] + w[1] * data[i].x[1] + b;
        
        // Prediction sign determines classification outcome
        int predicted_label = (score >= 0.0) ? 1 : -1;

        if (predicted_label == data[i].label) {
            correct_predictions++;
        }

        // Functional margin check: y * (w . x + b) < 1 means it is inside the margin street
        double functional_margin = data[i].label * score;
        if (functional_margin < 1.0) {
            margin_violations++;
        }
    }

    double accuracy = ((double)correct_predictions / NUM_POINTS) * 100.0;

    printf("--- Evaluation for %s ---\n", model_name);
    printf("Accuracy:               %.2f%% (%d/%d points)\n", accuracy, correct_predictions, NUM_POINTS);
    printf("Margin Violations:      %d points (points inside/on wrong side of margin boundary)\n\n", margin_violations);
}

int main() {
    srand(time(NULL));
    Point data[NUM_POINTS];

    // 1. Generate Synthetic Data separated by the line: x1 = 2*x0 + 1
    for (int i = 0; i < NUM_POINTS; i++) {
        data[i].x[0] = ((double)rand() / RAND_MAX) * 10 - 5; 
        data[i].x[1] = ((double)rand() / RAND_MAX) * 10 - 5; 
        
        double line_val = 2 * data[i].x[0] + 1;
        double noise = (((double)rand() / RAND_MAX) * 2.0) - 1.0; 
        
        if (data[i].x[1] > line_val + noise) {
            data[i].label = 1;
        } else {
            data[i].label = -1;
        }
    }

    double soft_w[2], soft_b;
    double max_w[2], max_b;

    // 2. Train Models
    double C_soft = 9.0;
    train_svm(data, C_soft, soft_w, &soft_b);

    double C_max = 100000.0;
    train_svm(data, C_max, max_w, &max_b);

    // 3. Display Results Summary
    system("cls");
    
    printf("==================================================\n");
    printf("               SVM MODEL METRICS                  \n");
    printf("==================================================\n");
    printf("Original Target Line:      x1 = 2.00x + 1.00\n\n");

    printf("--- Model Parameters ---\n");
    printf("Soft Margin Line:          x1 = %.2fx + %.2f\n", -soft_w[0]/soft_w[1], -soft_b/soft_w[1]);
    printf("Max Margin Line:           x1 = %.2fx + %.2f\n\n", -max_w[0]/max_w[1], -max_b/max_w[1]);

    // 4. Run Evaluation Tool
    evaluate_model(data, soft_w, soft_b, "Soft Margin Model (C=9.0)");
    evaluate_model(data, max_w, max_b, "Max Margin Model (C=100000.0)");
    printf("==================================================\n");

    return 0;
}