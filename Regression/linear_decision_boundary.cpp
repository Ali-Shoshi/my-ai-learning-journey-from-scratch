    #include <iostream>
    #include <vector>
    #include <cstdlib>
    #include <ctime>
/*
 * Machine Learning Profile:
 * ----------------------------------------------------------------------
 * - Model: Single-Layer Perceptron (Linear Classifier / Binary Threshold Model)
 * - Optimizer: Perceptron Learning Rule (Fixed Learning Rate)
 * - Batch Method: Stochastic Online Learning (Updates weights immediately after checking each data point)
 * - Regularization: None (Unpenalized weights)
 * - Loss Math: Error-driven step adjustment Delta Rule: alpha * (target - prediction) * input
 * - Metric: Classification Accuracy (%) checked over a separate validation set
 * ----------------------------------------------------------------------
 * This program implements a single-layer Perceptron from scratch to learn a binary classification 
 * boundary. It generates 10,000 synthetic data points where the target label (y) is 1 if the sum of 
 * two random integers exceeds 100, and 0 otherwise. The network models a linear decision boundary 
 * using a bias weight (w1) and two feature weights (w2, w3) paired with a step activation function. 
 * Over 40 epochs, the training loop applies the classic Perceptron Learning Rule to immediately 
 * correct parameters whenever a misclassification occurs, driving the system to discover the mathematical 
 * separation line between the two decision spaces.
 */
    class  Perceptron{
        public:
            double w1, w2 ,w3 ;
            double alpha=0.0005;
            Perceptron(double w1=0,double w2=0,double w3=0):w1(w1),w2(w2),w3(w3){}
            void printWeights() {
                std::cout << "Weights of the trained model: ";
                std::cout << "w1 = " << w1 << ", w2 = " << w2 << ", w3 = " << w3 << "\n\n";
            }
    };
    class DataPoint{
        public:
            int x1, x2;
            int y;
            DataPoint(int x1 ,int x2, int y): x1(x1) , x2(x2) , y(y){}
    };
    class PerceptronTrainer{
        private:
            Perceptron& perceptron;
            std::vector<DataPoint> datapoints ;
            void createDataPoints(){
                for ( int i =0; i < 10000 ; i++){
                    int rand1 = std::rand() % 100 + 1; 
                    int rand2 = std::rand() % 100 + 1; 
                    int y=((rand1 +rand2)>100) ? 1 : 0 ; 
                    datapoints.push_back(DataPoint(rand1, rand2,y));
                }
            }
            void train(int epochs){
                createDataPoints();
                for (int e = 0; e < epochs; ++e) {
                    for(const DataPoint& a : datapoints){
                        int y =predict(a.x1,a.x2);
                        
                            perceptron.w1+=perceptron.alpha*(a.y-y)*1;
                            perceptron.w2+=perceptron.alpha*(a.y-y)*a.x1;
                            perceptron.w3+=perceptron.alpha*(a.y-y)*a.x2;
                        
                    }
                }
            }
        public : 
            int predict(int x1 ,int x2){
                double y= perceptron.w1 + perceptron.w2*x1 + perceptron.w3*x2;
                int result=(y >= 0) ? 1 : 0;
                return result;
            }
            void evaluate(int numSamples = 1000) {
                int correct = 0;

                for (int i = 0; i < numSamples; ++i) {
                    int x1 = std::rand() % 100 + 1;
                    int x2 = std::rand() % 100 + 1;
                    int actual = (x1 + x2 > 100) ? 1 : 0;
                    int predicted = predict(x1, x2);

                    if (predicted == actual) {
                        correct++;
                    }
                }
                double accuracy = (double)correct / numSamples * 100;

                std::cout<<"The model is trying to figure out if x1 + x2 > 100\n";
                std::cout << "Accuracy on " << numSamples << " test samples: " << accuracy << "%\n";
            }
            PerceptronTrainer(Perceptron &ai): perceptron(ai){}

            void run(){
                train(40);  
            }
    };
    int main(){
        system("cls");
        std::srand(std::time(nullptr));  
        Perceptron perceptron(0,0,0);
        PerceptronTrainer perceptronTrainer(perceptron);
        perceptronTrainer.run();
        perceptronTrainer.evaluate();
        perceptron.printWeights();
        return 0 ;
    }