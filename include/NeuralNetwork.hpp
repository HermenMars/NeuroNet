#pragma once

#include "Matrix.hpp"

#include <vector>
#include <string>

class NeuralNetwork {
public:
    enum class Activation {
        ReLU,
        Sigmoid
    };

    explicit NeuralNetwork(
        const std::vector<std::size_t>& architecture
    );

    std::vector<double> predict(
        const std::vector<double>& input
    ) const;

    void train(
        const std::vector<double>& input,
        const std::vector<double>& target,
        double learningRate
    );

    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    std::vector<std::size_t> architecture;

    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

    double activate(double x, Activation function) const;
    double derivative(double x, Activation function) const;

    std::vector<double> forward(
        const std::vector<double>& input,
        std::vector<std::vector<double>>& activations,
        std::vector<std::vector<double>>& zValues
    ) const;
};