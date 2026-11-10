#include "NeuralNetwork.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(
    const std::vector<std::size_t>& layers
) : architecture(layers) {

    if (layers.size() < 2)
        throw std::invalid_argument("Network needs at least two layers");

    for (std::size_t size : layers) {
        if (size == 0)
            throw std::invalid_argument("Layer size cannot be zero");
    }

    for (std::size_t i = 0; i + 1 < layers.size(); ++i) {
        weights.emplace_back(layers[i + 1], layers[i]);
        biases.emplace_back(layers[i + 1], 1);

        weights.back().randomize();
        biases.back().fill(0.0);
    }
}

double NeuralNetwork::activate(
    double x,
    Activation function
) const {
    if (function == Activation::ReLU)
        return x > 0.0 ? x : 0.0;

    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::derivative(
    double x,
    Activation function
) const {
    if (function == Activation::ReLU)
        return x > 0.0 ? 1.0 : 0.0;

    double s = activate(x, Activation::Sigmoid);
    return s * (1.0 - s);
}

std::vector<double> NeuralNetwork::forward(
    const std::vector<double>& input,
    std::vector<std::vector<double>>& activations,
    std::vector<std::vector<double>>& zValues
) const {

    if (input.size() != architecture.front())
        throw std::invalid_argument("Incorrect input size");

    activations.clear();
    zValues.clear();

    activations.push_back(input);

    for (std::size_t layer = 0; layer < weights.size(); ++layer) {
        std::vector<double> z(architecture[layer + 1]);
        std::vector<double> output(architecture[layer + 1]);

        Activation function =
            (layer == weights.size() - 1)
            ? Activation::Sigmoid
            : Activation::ReLU;

        for (std::size_t neuron = 0;
             neuron < architecture[layer + 1];
             ++neuron) {

            double sum = biases[layer](neuron, 0);

            for (std::size_t i = 0;
                 i < architecture[layer];
                 ++i) {

                sum += weights[layer](neuron, i)
                     * activations.back()[i];
            }

            z[neuron] = sum;
            output[neuron] = activate(sum, function);
        }

        zValues.push_back(z);
        activations.push_back(output);
    }

    return activations.back();
}

std::vector<double> NeuralNetwork::predict(
    const std::vector<double>& input
) const {

    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> zValues;

    return forward(input, activations, zValues);
}

void NeuralNetwork::train(
    const std::vector<double>& input,
    const std::vector<double>& target,
    double learningRate
) {

    if (target.size() != architecture.back())
        throw std::invalid_argument("Incorrect target size");

    if (learningRate <= 0.0)
        throw std::invalid_argument("Learning rate must be positive");

    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> zValues;

    forward(input, activations, zValues);

    std::vector<std::vector<double>> deltas(weights.size());

    std::size_t last = weights.size() - 1;

    // Output layer delta
    deltas[last].resize(architecture.back());

    for (std::size_t i = 0; i < architecture.back(); ++i) {
        double error = activations.back()[i] - target[i];

        deltas[last][i] =
            error * derivative(
                zValues[last][i],
                Activation::Sigmoid
            );
    }

    // Hidden layer deltas
    for (int layer = static_cast<int>(last) - 1;
         layer >= 0;
         --layer) {

        deltas[layer].resize(architecture[layer + 1]);

        for (std::size_t i = 0;
             i < architecture[layer + 1];
             ++i) {

            double error = 0.0;

            for (std::size_t j = 0;
                 j < architecture[layer + 2];
                 ++j) {

                error += weights[layer + 1](j, i)
                       * deltas[layer + 1][j];
            }

            deltas[layer][i] =
                error * derivative(
                    zValues[layer][i],
                    Activation::ReLU
                );
        }
    }

    // Update weights and biases
    for (std::size_t layer = 0;
         layer < weights.size();
         ++layer) {

        for (std::size_t i = 0;
             i < architecture[layer + 1];
             ++i) {

            biases[layer](i, 0) -=
                learningRate * deltas[layer][i];

            for (std::size_t j = 0;
                 j < architecture[layer];
                 ++j) {

                weights[layer](i, j) -=
                    learningRate *
                    deltas[layer][i] *
                    activations[layer][j];
            }
        }
    }
}

void NeuralNetwork::save(const std::string& filename) const {
    std::ofstream file(filename);

    if (!file)
        throw std::runtime_error("Cannot save model");

    file << architecture.size() << '\n';

    for (auto size : architecture)
        file << size << ' ';

    file << '\n';

    for (std::size_t layer = 0; layer < weights.size(); ++layer) {
        for (double value : weights[layer].data)
            file << value << ' ';

        file << '\n';

        for (double value : biases[layer].data)
            file << value << ' ';

        file << '\n';
    }
}

void NeuralNetwork::load(const std::string& filename) {
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error("Cannot load model");

    std::size_t layerCount;
    file >> layerCount;

    if (layerCount != architecture.size())
        throw std::runtime_error("Architecture mismatch");

    for (std::size_t i = 0; i < layerCount; ++i) {
        std::size_t size;
        file >> size;

        if (size != architecture[i])
            throw std::runtime_error("Architecture mismatch");
    }

    for (std::size_t layer = 0; layer < weights.size(); ++layer) {
        for (double& value : weights[layer].data)
            file >> value;

        for (double& value : biases[layer].data)
            file >> value;
    }

    if (!file)
        throw std::runtime_error("Invalid model file");
}