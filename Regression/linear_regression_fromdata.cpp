#include <iostream>

/*
 * Machine Learning Profile:
 * ----------------------------------------------------------------------
 * - Model: Simple Linear Regression (Single Variable Feature Mapping Model)
 * - Optimizer: Stochastic Gradient Descent (SGD with Split Learning Rates)
 * - Batch Method: Online Learning (Updates weights immediately after evaluating each sample)
 * - Regularization: None (Unpenalized weights)
 * - Loss Math: Mean Squared Error Loss (MSE): sum((prediction - target)^2) / N
 * - Metric: Mean Squared Error (MSE) logged per epoch
 * ----------------------------------------------------------------------
 * This program implements a Simple Linear Regression model from scratch in C++ to predict 
 * numeric targets using a single input feature. It trains on a static dataset of 30 paired 
 * data points modeling a linear trend (y = wx + b). Starting with initial weights (w = -5, b = 100), 
 * the network uses continuous linear activation to calculate predictions. Over 150 epochs, 
 * the training loop applies gradient descent to calculate partial derivatives (derivativeW, derivativeB) 
 * for each sample, immediately updating the weight parameter (w) and bias parameter (b) using differential 
 * learning rates (0.00001 for w, 0.01 for b) to incrementally minimize squared error loss.
 */

class Model{
    public:
        double w,b;
        Model(double w = -5, double b = 100) : w(w), b(b){}

        double predict(double x){
            return w * x + b;
        }

        void train(){
            int dataset[][2] = {
                {35, 54},{38, 58},{40, 62},{42, 60},{45, 68},{48, 72},{50, 71},{53, 78},{55, 80},{58, 81},
                {60, 85},{62, 88},{65, 90},{68, 92},{70, 97},{72, 99},{75, 102},{78, 106},{80, 105},{82, 110},
                {85, 112},{88, 118},{90, 120},{93, 122},{95, 126},{98, 129},{100, 131},{105, 138},{110, 142}, {115, 148}
            };

            int n = sizeof(dataset) / sizeof(dataset[0]);
            double learningRateW = 0.00001;
            double learningRateB = 0.01;


            for ( int epoch = 0; epoch < 150; ++epoch) {
                double totalError = 0.0;
                double derivativeW;
                double derivativeB;

                for (int i = 0; i < n; ++i) {
                    double x = dataset[i][0];
                    double y = dataset[i][1];
                    double prediction = predict(x);
                    double error = prediction - y;

                    totalError += error * error;
                    derivativeW = error * x;
                    derivativeB = error;

                    w -= learningRateW * 2.0 * derivativeW;
                    b -= learningRateB * 2.0 * derivativeB;
                }

                totalError /= n;
                std::cout << "Epoch " << epoch << ", MSE: " << totalError;
                std::cout << "  |  w = " << w << ", b = " << b << "\n";

            }
        }
        
        
        void printWeights(){
            std::cout << " Weights : ";
            std::cout << "w = " << w << ", b = " << b << "\n";
        }
        
};


int main(){

    Model model;


    model.printWeights();
    model.train();
    model.printWeights();

    return 0;

}