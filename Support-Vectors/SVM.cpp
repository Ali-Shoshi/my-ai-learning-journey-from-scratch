#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <ctime>
#include <cstdlib>

struct DataPoint {
    int positionx;
    int positiony;
    int label;

    DataPoint(int positionx, int positiony, int lable) 
        : positionx(positionx), positiony(positiony), label(lable) {}
};

class SupportVectorMachine {
    private:
        std::vector<DataPoint> dataset;
        std::vector<DataPoint> testingdataset;
        
        // Hyperparameters optimized for the coordinate system grid
        double gamma = 0.05; 
        double lambda = 0.01; 
        std::vector<double> alpha;
        int stepCounter = 0;

        double rbfKernel(DataPoint dp1, DataPoint dp2) const {
            double sq_dist = std::pow(dp1.positionx - dp2.positionx, 2) + 
                             std::pow(dp1.positiony - dp2.positiony, 2);
            return std::exp(-gamma * sq_dist);
        }

    public:
        void createData() {
            dataset.clear();
            testingdataset.clear();
            alpha.clear();

            // Dataset ranges scaled to [-15, 15] to properly populate a radius 10 circle (100)
            for(int i = 0; i < 600; i++) {
                int positionx = std::rand() % 30 - 15; 
                int positiony = std::rand() % 30 - 15; 
                int lable = (std::pow(positionx, 2) + std::pow(positiony, 2) <= 100) ? 1 : -1;
                dataset.push_back(DataPoint(positionx, positiony, lable));
            }
            
            for(int i = 0; i < 200; i++) {
                int positionx = std::rand() % 30 - 15; 
                int positiony = std::rand() % 30 - 15; 
                int lable = (std::pow(positionx, 2) + std::pow(positiony, 2) <= 100) ? 1 : -1;
                testingdataset.push_back(DataPoint(positionx, positiony, lable));
            }
            
            for(size_t i = 0; i < dataset.size(); i++) {
                alpha.push_back(0.0);
            }
        }

        void fit(int epochs = 20) {
            stepCounter = 0; 

            for (int epoch = 0; epoch < epochs; epoch++) {
                for (size_t i = 0; i < dataset.size(); i++) {
                    stepCounter++; 
                    
                    double rawPredictedScore = 0.0;
                    for (size_t j = 0; j < dataset.size(); j++) {
                        if (alpha[j] > 0.0) {
                            rawPredictedScore += alpha[j] * dataset[j].label * rbfKernel(dataset[j], dataset[i]);
                        }
                    }
                    
                    if (dataset[i].label * rawPredictedScore < 1.0) {
                        alpha[i] += 1.0 / (lambda * stepCounter);
                    }
                }
            }
        }

        int predict(const DataPoint query) const {
            double score = 0.0;
            for (size_t j = 0; j < dataset.size(); ++j) {
                if (alpha[j] > 0.0) {
                    score += alpha[j] * dataset[j].label * rbfKernel(dataset[j], query);
                }
            }
            return (score >= 0.0) ? 1 : -1;
        }

        void evaluate() const {
            int correct = 0;
            for (const auto& testPoint : testingdataset) {
                if (predict(testPoint) == testPoint.label) {
                    correct++;
                }
            }
            double accuracy = (static_cast<double>(correct) / testingdataset.size()) * 100.0;
            std::cout << "Model Evaluation Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%\n";
            
            int sv_count = 0;
            for(double a : alpha) {
                if (a > 1e-6) sv_count++;
            }
            std::cout << "Identified Support Vectors holding the boundary: " << sv_count << "\n";
        }
};

int main() {
    std::system("cls");
    std::srand(std::time(nullptr));

    SupportVectorMachine svm;
    
    std::cout << "=== Running Custom Soft-Margin RBF SVM ===\n";
    std::cout << "[1/3] Generating Concentric Datasets...\n";
    svm.createData();
    
    std::cout << "[2/3] Executing Pegasos Optimization Solver Loops...\n";
    svm.fit(25); // Train for 25 complete dataset sweeps
    
    std::cout << "[3/3] Analyzing Boundary Success Metrics...\n";
    svm.evaluate();
    std::cout << "==========================================\n";

    return 0;
}