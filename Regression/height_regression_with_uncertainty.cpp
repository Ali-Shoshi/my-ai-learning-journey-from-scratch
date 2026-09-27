#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

struct DataPoint {
    double age;
    double is_male; // 0 = female, 1 = male
    double height;  // centimetres
};

double relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

double average_height(double age, bool is_male) {
    constexpr std::array<double, 10> ages{
        0, 1, 5, 10, 15, 18, 30, 60, 80, 100
    };

    constexpr std::array<double, 10> female_heights{
        50, 75, 109, 138, 161, 164, 164, 163, 160, 157
    };

    constexpr std::array<double, 10> male_heights{
        51, 76, 110, 140, 169, 177, 177, 176, 172, 169
    };

    const auto& heights = is_male ? male_heights : female_heights;

    if (age <= ages.front()) {
        return heights.front();
    }

    for (std::size_t i = 1; i < ages.size(); ++i) {
        if (age <= ages[i]) {
            const double fraction =
                (age - ages[i - 1]) / (ages[i] - ages[i - 1]);

            return heights[i - 1] +
                   fraction * (heights[i] - heights[i - 1]);
        }
    }

    return heights.back();
}

// The synthetic standard deviation changes with age.
double synthetic_height_sd(double age) {
    const double puberty =
        std::exp(-std::pow((age - 14.0) / 5.0, 2.0));

    const double older_age =
        std::max(0.0, (age - 70.0) / 30.0);

    return 2.0
         + 4.5 * (1.0 - std::exp(-age / 10.0))
         + 2.0 * puberty
         + 1.2 * older_age;
}

std::vector<DataPoint> make_dataset(std::size_t count) {
    std::mt19937 rng(42); // Fixed seed, so the same data is generated each run.

    std::uniform_real_distribution<double> age_distribution(0.0, 100.0);
    std::bernoulli_distribution sex_distribution(0.5);
    std::normal_distribution<double> standard_normal(0.0, 1.0);

    std::vector<DataPoint> points;
    points.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        const double age = age_distribution(rng);
        const bool is_male = sex_distribution(rng);

        const double mean_height = average_height(age, is_male);
        const double sd = synthetic_height_sd(age);
        const double height = mean_height + sd * standard_normal(rng);

        points.push_back({
            age,
            is_male ? 1.0 : 0.0,
            height
        });
    }

    return points;
}

class Model {
public:
    // Network dimensions: 2 inputs, 4 hidden neurons, 2 outputs.
    std::vector<std::vector<double>> W_I_H;
    std::vector<std::vector<double>> W_H_O;
    std::vector<double> b_I_H;
    std::vector<double> b_H_O;

    std::vector<double> hidden;
    std::vector<double> output;
    std::vector<DataPoint> dataset;

    double learning_rate = 0.005;

    // Used to normalize heights while training.
    double target_mean = 0.0;
    double target_sd = 1.0;

    Model()
        : W_I_H(2, std::vector<double>(4, 0.0)),
          W_H_O(4, std::vector<double>(2, 0.0)),
          b_I_H(4, 0.0),
          b_H_O(2, 0.0),
          hidden(4, 0.0),
          output(2, 0.0) {
        std::mt19937 rng(7);

        // He-style initialization for the ReLU hidden layer.
        std::normal_distribution<double> hidden_init(
            0.0, std::sqrt(2.0 / 2.0)
        );

        for (int input = 0; input < 2; ++input) {
            for (int h = 0; h < 4; ++h) {
                W_I_H[input][h] = hidden_init(rng);
            }
        }

        // Small initial weights for the linear output layer.
        std::normal_distribution<double> output_init(0.0, 0.01);

        for (int h = 0; h < 4; ++h) {
            for (int o = 0; o < 2; ++o) {
                W_H_O[h][o] = output_init(rng);
            }
        }
    }

    void forward(const double input[2], double z_hidden[4]) {
        // Input layer to hidden layer, then ReLU.
        for (int h = 0; h < 4; ++h) {
            z_hidden[h] = b_I_H[h];

            for (int j = 0; j < 2; ++j) {
                z_hidden[h] += W_I_H[j][h] * input[j];
            }

            hidden[h] = relu(z_hidden[h]);
        }

        // Linear output layer.
        // output[0] = predicted normalized mean height
        // output[1] = predicted log variance
        for (int o = 0; o < 2; ++o) {
            output[o] = b_H_O[o];

            for (int h = 0; h < 4; ++h) {
                output[o] += W_H_O[h][o] * hidden[h];
            }
        }
    }

    void train(int epochs = 2000, std::size_t batch_size = 30) {
        if (dataset.empty()) {
            std::cout << "Dataset is empty.\n";
            return;
        }

        if (batch_size == 0) {
            std::cout << "Batch size must be greater than zero.\n";
            return;
        }

        // Calculate height mean and standard deviation for normalization.
        target_mean = 0.0;
        for (const DataPoint& point : dataset) {
            target_mean += point.height;
        }
        target_mean /= static_cast<double>(dataset.size());

        double variance = 0.0;
        for (const DataPoint& point : dataset) {
            const double difference = point.height - target_mean;
            variance += difference * difference;
        }
        variance /= static_cast<double>(dataset.size());

        target_sd = std::sqrt(variance);
        if (target_sd < 1e-8) {
            target_sd = 1.0;
        }

        std::vector<std::size_t> order(dataset.size());
        std::iota(order.begin(), order.end(), 0);

        std::mt19937 shuffle_rng(123);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), shuffle_rng);
            double epoch_loss = 0.0;

            for (std::size_t start = 0; start < order.size();
                 start += batch_size) {
                const std::size_t end =
                    std::min(start + batch_size, order.size());
                const double batch_count =
                    static_cast<double>(end - start);

                double g_W_I_H[2][4]{};
                double g_W_H_O[4][2]{};
                double g_b_I_H[4]{};
                double g_b_H_O[2]{};

                for (std::size_t position = start;
                     position < end;
                     ++position) {
                    const DataPoint& point = dataset[order[position]];

                    // Scale age to 0..1 and encode sex as 0 or 1.
                    const double inputs[2]{
                        point.age / 100.0,
                        point.is_male
                    };

                    double z_hidden[4];
                    forward(inputs, z_hidden);

                    // Train using normalized height.
                    const double y =
                        (point.height - target_mean) / target_sd;

                    const double mu = output[0];
                    const double log_variance = output[1];
                    const double error = y - mu;
                    const double inverse_variance =
                        std::exp(-log_variance);

                    // Gaussian negative log-likelihood.
                    epoch_loss += 0.5 *
                        (error * error * inverse_variance + log_variance);

                    // Derivatives for the two outputs.
                    const double d_mu =
                        (mu - y) * inverse_variance;

                    const double d_log_variance =
                        0.5 * (1.0 -
                               error * error * inverse_variance);

                    for (int h = 0; h < 4; ++h) {
                        g_W_H_O[h][0] += hidden[h] * d_mu;
                        g_W_H_O[h][1] += hidden[h] * d_log_variance;

                        // Backpropagate through the output weights and ReLU.
                        const double relu_gradient =
                            (z_hidden[h] > 0.0) ? 1.0 : 0.0;

                        const double d_hidden =
                            (W_H_O[h][0] * d_mu +
                             W_H_O[h][1] * d_log_variance)
                            * relu_gradient;

                        for (int j = 0; j < 2; ++j) {
                            g_W_I_H[j][h] += inputs[j] * d_hidden;
                        }

                        g_b_I_H[h] += d_hidden;
                    }

                    g_b_H_O[0] += d_mu;
                    g_b_H_O[1] += d_log_variance;
                }

                // Apply the average gradients for this mini-batch.
                for (int j = 0; j < 2; ++j) {
                    for (int h = 0; h < 4; ++h) {
                        W_I_H[j][h] -=
                            learning_rate * g_W_I_H[j][h] / batch_count;
                    }
                }

                for (int h = 0; h < 4; ++h) {
                    for (int o = 0; o < 2; ++o) {
                        W_H_O[h][o] -=
                            learning_rate * g_W_H_O[h][o] / batch_count;
                    }

                    b_I_H[h] -=
                        learning_rate * g_b_I_H[h] / batch_count;
                }

                b_H_O[0] -=
                    learning_rate * g_b_H_O[0] / batch_count;
                b_H_O[1] -=
                    learning_rate * g_b_H_O[1] / batch_count;
            }

            if (epoch == 0 || (epoch + 1) % 100 == 0) {
                std::cout << "Epoch " << (epoch + 1)
                          << ", average loss = "
                          << epoch_loss / dataset.size() << '\n';
            }
        }
    }

    void predict(double age, bool is_male) {
        const double inputs[2]{
            age / 100.0,
            is_male ? 1.0 : 0.0
        };

        double z_hidden[4];
        forward(inputs, z_hidden);

        const double predicted_mean =
            target_mean + target_sd * output[0];

        const double predicted_sd =
            target_sd * std::exp(0.5 * output[1]);

        std::cout << (is_male ? "Male" : "Female")
                  << ", age " << age
                  << ": predicted height = " << predicted_mean
                  << " cm, predicted standard deviation = "
                  << predicted_sd << " cm\n";
    }
};

int main() {
    Model model;

    // Generate exactly 150 synthetic data points.
    model.dataset = make_dataset(150);

    std::cout << "Generated " << model.dataset.size()
              << " synthetic data points.\n\n";

    model.train();

    std::cout << "\nExample predictions:\n";
    model.predict(5, false);
    model.predict(15, false);
    model.predict(30, false);
    model.predict(80, false);
    model.predict(5, true);
    model.predict(15, true);
    model.predict(30, true);
    model.predict(80, true);

    return 0;
}