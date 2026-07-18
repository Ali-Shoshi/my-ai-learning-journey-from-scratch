#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

/*
 * Machine Learning Profile:
 * ----------------------------------------------------------------------
 * - Model: Polynomial Logistic Regression (Quadratic / Binary Probability Model)
 * - Optimizer: Gradient Descent / Delta Rule (Fixed Learning Rate)
 * - Batch Method: Stochastic Online Learning (Updates weights sequentially per data point)
 * - Regularization: None (Unpenalized weights)
 * - Loss Math: Cross-Entropy derived error step adjustment: alpha * (target - probability) * input
 * - Activation: Sigmoid Function maps real values to a probability distribution [0, 1]
 * - Metric: Classification Accuracy (%) measured over a random validation threshold (>= 0.5)
 * ----------------------------------------------------------------------
 * This program implements a Polynomial Logistic Regression model from scratch using a 
 * quadratic hypothesis to map a probability curve for daytime vs. nighttime hours. It 
 * generates 10,000 synthetic timestamps where the target label (isLight) is true (1) 
 * if the hour falls within a designated daylight window between sunrise (6.00) and 
 * sunset (18.00), and false (0) otherwise. The system normalizes time features into a 
 * continuous range [0, 1] and applies polynomial expansion to generate a quadratic feature 
 * (hour^2). Using a bias weight (w0), a linear weight (w1), and a quadratic weight (w2) 
 * passed through a sigmoid activation, the system tracks a parabolic decision boundary. 
 * Over 100 epochs, the network updates parameters via stochastic gradient descent, 
 * effectively learning to bend its decision space to capture both thresholds simultaneously.
 */

class DayNightModel{
    public:
        double w0, w1, w2;
        double alpha = 0.01;

        DayNightModel(double w0 = 0.0, double w1 = 0.0, double w2 = 0.0) : w0(w0), w1(w1), w2(w2) {}

        void printWeights() {
            std::cout << "\nWeights : ";
            std::cout << "w0 = " << w0 << ", w1 = " << w1 << ", w2 = " << w2 << "\n";
        }
};

class DataPoint {
    public:
        double hour;
        bool isLight;

        DataPoint(double hour, bool isLight) : hour(hour), isLight(isLight) {}
};

class ModelTrainer {
    private:
        DayNightModel& dayNightModel;
        std::vector<DataPoint> datapoints;
        double sunriseHour = 6.0;
        double sunsetHour = 18.0;


        double sigmoid(double x) {
            return 1.0 / (1.0 + std::exp(-x));
        }

        void createDataPoints() {
            for (int i = 0; i < 10000; ++i) {
                double hour = (std::rand() % 2400) / 100.0; 
                bool isLight = (hour <= sunsetHour && hour >= sunriseHour) ? 1 : 0;
                datapoints.emplace_back(hour, isLight);
            }
        }

        void train(int epochs) {
            createDataPoints();
            for (int e = 0; e < epochs; ++e) {
                for (const DataPoint& dp : datapoints) {
                    double normalize=dp.hour/24.0;
                    double z = dayNightModel.w0 + dayNightModel.w1 * normalize + dayNightModel.w2 * std::pow(normalize,2);
                    double predicted = sigmoid(z);
                    double error = predicted - dp.isLight ;

                    dayNightModel.w0 -= dayNightModel.alpha * error;
                    dayNightModel.w1 -= dayNightModel.alpha * error * normalize;
                    dayNightModel.w2 -= dayNightModel.alpha * error * std::pow(normalize,2);

                }
            }
        }

    public:
        ModelTrainer(DayNightModel& dayNightModel) : dayNightModel(dayNightModel) {}

        double predict(double hour) {
            double z = dayNightModel.w0 + dayNightModel.w1 * hour/24.0 + dayNightModel.w2 * std::pow(hour/24.0,2);
            return sigmoid(z); 
        }

        void test(double hour) {
            double probability = predict(hour);
            std::cout << "Hour: " << hour << " , Light probability: " << probability;
            std::cout << " , Prediction: " << ((probability >= 0.5) ? "Light\n" : "Dark\n");
        }

        void evaluate(int numSamples = 1000) {
            int correct = 0;
            for (int i = 0; i < numSamples; ++i) {
                double hour = std::rand() % 24;
                bool actual = (hour <= sunsetHour && hour >= sunriseHour) ? 1 : 0;
                double predicted = predict(hour);
                bool result = predicted >= 0.5;
                if (result == actual) correct++;
            }
            double accuracy = (double)correct / numSamples * 100.0;
            std::cout << "Accuracy on " << numSamples << " test samples: " << accuracy << "%\n";
        }

        void run() {
            train(100);
            std::cout<<"Sunrise hour : "<<sunriseHour <<", Sunset hour : "<<sunsetHour<<"\n\n";
            std::cout<<"Predictions of the model : \n";
            test(4);
            test(5);
            test(6);
            test(7);
            test(12);
            test(17);
            test(18);
            test(19);
            test(20);

        }
};

int main() {
    system("cls");
    std::srand(std::time(nullptr));

    DayNightModel dayNightModel;
    ModelTrainer modelTrainer(dayNightModel);
    modelTrainer.run();
    dayNightModel.printWeights();
    modelTrainer.evaluate();
    return 0;
}
