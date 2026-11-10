#include "NeuralNetwork.hpp"

#include <iostream>
#include <vector>
#include <iomanip>

int main() {

    // Network architecture: 2 -> 8 -> 1
    NeuralNetwork nn({2, 8, 1});

    // XOR training dataset
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    const int epochs = 20000;
    const double learningRate = 0.05;

    std::cout << "Training neural network...\n";

    for (int epoch = 0; epoch < epochs; ++epoch) {

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            nn.train(inputs[i], targets[i], learningRate);
        }

        if ((epoch + 1) % 2000 == 0) {
            std::cout << "Epoch: " << epoch + 1 << '\n';
        }
    }

    std::cout << "\nTraining complete!\n\n";

    // Test predictions
    std::cout << std::fixed << std::setprecision(4);

    for (const auto& input : inputs) {

        auto output = nn.predict(input);

        std::cout
            << input[0] << " XOR " << input[1]
            << " = " << output[0] << '\n';
    }

    // Save trained model
    nn.save("model.nn");

    std::cout << "\nModel saved to model.nn\n";

    return 0;
}