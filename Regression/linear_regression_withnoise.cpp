#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
/*
 * Machine Learning Profile:
 * ----------------------------------------------------------------------
 * - Model: Linear Regression (Linear Model)
 * - Optimizer: Vanilla Gradient Descent (Fixed Learning Rate)
 * - Batch Method: Batch Gradient Descent (Updates weights once per epoch using all data)
 * - Regularization: L2 Regularization (Weight Decay)
 * - Loss Math: Mean Squared Error (MSE) derivatives used for training
 * - Metric: Mean Absolute Error (MAE) used for evaluation  
 * ----------------------------------------------------------------------
 * This program implements a simple linear regression model from scratch using Vanilla 
 * Gradient Descent with integrated L2 Regularization (Weight Decay). It generates 10,000 
 * synthetic datapoints around a hidden baseline equation (y = 52x + 13) injected with random 
 * noise, then iteratively trains the model's parameters (w0 and w1) over 500 epochs. By 
 * balancing data error minimization with an active weight complexity penalty (lambda), the 
 * training loop smoothly guides the parameters toward the ground truth values while 
 * actively preventing overfitting to the generated noise.
 */

class Model{
    public:
        double w0, w1;
        Model(double w0 = 0.5, double w1 = 0.5) : w0(w0), w1(w1){}

        void printWeights(){
            std::cout << " Weights : ";
            std::cout << "w0 = " << w0 << ", w1 = " << w1 << "\n";
        }

        double predict(double x){
            return w0 * x + w1;
        }
};

class DataPoint{
    public:
        double x;
        double y;
        DataPoint(double x, double y) : x(x), y(y){}
};

class ModelTrainer{
    private: 
        Model& model;
        std::vector<DataPoint> dataPoints;
        double alpha = 0.05; 
        double lambda = 0.001;
        double meanAbsoluteError = 0;

        void createDataPoints(){
            for (int i = 0; i < 10000; i++){
                double rand1 = -1.0 + (std::rand() / (RAND_MAX + 1.0)) * 2.0;
                double temp = std::rand() % 11 - 5; // Noise between -5 and 5
                double y = 52 * rand1 + 13;         // Ground truth equation
                y += temp;                         // add noise
                dataPoints.push_back(DataPoint(rand1, y));
            }
        }

        void train(int epoch){
            double numberOfDatapoints = static_cast<double>(dataPoints.size()); 
            
            for (int round = 0; round < epoch; round++){
                double sumW0 = 0;
                double sumW1 = 0;
                
                for (auto & dp : dataPoints){
                    double predictdy = model.predict(dp.x);
                    double error = predictdy - dp.y;
                    
                    sumW0 += error * dp.x;
                    sumW1 += error;
                }

                model.w0 -= (alpha * (2.0 / numberOfDatapoints) * sumW0) + (alpha * lambda * model.w0); 
                model.w1 -= (alpha * (2.0 / numberOfDatapoints) * sumW1) + (alpha * lambda * model.w1); 
            }
        }

    public:
        ModelTrainer(Model& model) : model(model){}
        
        void evaluate(){
            double totalError = 0;
            for (auto& dp : dataPoints){
                totalError += std::abs(dp.y - model.predict(dp.x));
            }
            meanAbsoluteError = totalError / dataPoints.size();
        }

        void run(){
            createDataPoints();
            train(500); 
            evaluate();

            std::cout << "The equation from which the synthetic data was created is:\n";
            std::cout << "y = 52 * random + 13;\n";
            std::cout << "The weights of the synthetic data are: w0 = 52, w1 = 13\n";
            std::cout << "Injected random data noise range: [-5, 5]\n\n";

            std::cout << "The model's weights after training are: W0 = " << model.w0 << ", W1 = " << model.w1 << "\n";
            std::cout << "The Mean Absolute Error is: " << meanAbsoluteError << "\n";
        }
};

int main(){
    system("cls");
    std::srand(std::time(nullptr));
    
    Model model;
    ModelTrainer mt(model);
    mt.run();

    return 0;
}