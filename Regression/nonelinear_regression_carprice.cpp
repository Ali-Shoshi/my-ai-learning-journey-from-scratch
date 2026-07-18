#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

/*
 * Machine Learning Profile:
 * ----------------------------------------------------------------------
 * - Model: Exponential Regression (Non-Linear Model)
 * - Optimizer: Vanilla Gradient Descent (Fixed Learning Rate)
 * - Batch Method: Stochastic Gradient Descent / SGD (Updates weights immediately after each data point)
 * - Regularization: None (Unpenalized parameters)
 * - Loss Math: Mean Squared Error (MSE) custom exponential partial derivatives
 * - Metric: Mean Absolute Error (MAE) used for evaluation  
 * ----------------------------------------------------------------------
 * This program implements a non-linear exponential regression model from scratch to predict 
 * used car resale values based on three features: original purchase price, age, and mileage. 
 * The dataset is generated synthetically using feature scaling and a hidden, true underlying 
 * equation (w0=1.0, w1=3.2, w2=0.08). Training is performed over 100 epochs using Stochastic 
 * Gradient Descent (SGD) with a fixed learning rate (alpha = 0.05). By calculating the 
 * custom partial derivatives of the exponential function for each weight (w0, w1, w2) on 
 * every data point, the optimization loop shifts the model's parameters to fit the non-linear 
 * decay curves and minimizes prediction error.
 */

class CarPriceModel{
    public: 
        long double w0, w1, w2;
        
        CarPriceModel(long double w0 = 0.5, long double w1 = 0.5, long double w2 = 0.5)
        : w0(w0), w1(w1), w2(w2){}
        
        long double predictScaled(double price, double yearOld, double mileage){
            double price_scaled = price / 300000.0;
            double yearOld_scaled = yearOld / 30.0;
            double mileage_scaled = mileage / 300000.0;

            return w0 * price_scaled * std::exp(w1 * (-yearOld_scaled)) * std::exp(w2 * (-mileage_scaled));
        }
        
        long double predict(double price, double yearOld, double mileage){
            return predictScaled(price, yearOld, mileage) * 300000.0;
        }
};

class DataPoint{
    public: 
        double boughtPrice, yearOld, milage, estimatedPrice;
        DataPoint(double boughtPrice = 0, double yearOld = 0, double milage = 0, double estimatedPrice = 0)
        : boughtPrice(boughtPrice), yearOld(yearOld), milage(milage), estimatedPrice(estimatedPrice){}
};

class ModelTrainer{
    private: 
        CarPriceModel& carPriceModel;
        std::vector<DataPoint> dataPoints;
        double alpha = 0.05; 
        double meanAbsoluteError = 0;

        long double getRandomDouble(long double min, long double max){
            return min + (max - min) * (rand() / (double)RAND_MAX);
        }

        void createDataPoints(){
            long double w0_true = 1.0;
            long double w1_true = 3.2;
            long double w2_true = 0.08;

            for(int i = 0; i < 2000; i++){
                long double boughtPrice = getRandomDouble(10000, 300000);
                long double yearOld = getRandomDouble(0, 30);
                long double milage = getRandomDouble(0, 300000);

                long double estimatedPrice = w0_true * (boughtPrice / 300000.0) * std::exp(w1_true * (-yearOld / 30.0)) * std::exp(w2_true * (-milage / 300000.0));

                dataPoints.emplace_back(boughtPrice, yearOld, milage, estimatedPrice);
            }
        }

        void train(int epoch){
            for (int round = 0; round < epoch; round++){
                for (auto & dp : dataPoints){
                    double price_scaled = dp.boughtPrice / 300000.0;
                    double year_scaled = dp.yearOld / 30.0;
                    double mileage_scaled = dp.milage / 300000.0;

                    double predictedPriceScaled = carPriceModel.predictScaled(dp.boughtPrice, dp.yearOld, dp.milage);
                    double error = predictedPriceScaled - dp.estimatedPrice;

                    carPriceModel.w0 -= alpha * error * (price_scaled * std::exp(-carPriceModel.w1 * year_scaled) * std::exp(-carPriceModel.w2 * mileage_scaled));
                    carPriceModel.w1 -= alpha * error * (-year_scaled * predictedPriceScaled);
                    carPriceModel.w2 -= alpha * error * (-mileage_scaled * predictedPriceScaled);
                }
            }
        }

    public:
        ModelTrainer(CarPriceModel& carPriceModel) : carPriceModel(carPriceModel){}
        
        void evaluate(){
            double totalError = 0;
            for (auto& dp : dataPoints){
                totalError += std::abs((dp.estimatedPrice * 300000.0) - carPriceModel.predict(dp.boughtPrice, dp.yearOld, dp.milage));
            }
            meanAbsoluteError = totalError / dataPoints.size();
        }

        void run(){
            createDataPoints();
            train(100);
            evaluate();

            std::cout << "The equation from which the synthetic data was created is:\n";
            std::cout << "w0_true * boughtPrice/300000 * exp(w1_true * (-yearOld/30)) * exp(w2_true * (-mileage/300000))\n";
            std::cout << "The weights of the synthetic data are : ";
            std::cout << "w0 = 1 , w1 = 3.2 , w2 = 0.08\n\n";

            std::cout << "The model's weights after training are : W0 = " << carPriceModel.w0 << ", W1 = " << carPriceModel.w1 << ", W2 = " << carPriceModel.w2 << "\n";
            std::cout << "The Mean Absolute Error is: $" << meanAbsoluteError << "\n";
        }
};

int main(){
    system("cls");
    std::srand(std::time(nullptr));
    
    CarPriceModel carPriceModel;
    ModelTrainer mt(carPriceModel);
    mt.run();

    return 0;
}