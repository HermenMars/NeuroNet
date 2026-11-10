#include "Matrix.hpp"

#include <random>
#include <stdexcept>

Matrix::Matrix(std::size_t r, std::size_t c)
    : rows(r), cols(c), data(r * c, 0.0) {}

double& Matrix::operator()(std::size_t r, std::size_t c) {
    if (r >= rows || c >= cols)
        throw std::out_of_range("Matrix index out of range");

    return data[r * cols + c];
}

double Matrix::operator()(std::size_t r, std::size_t c) const {
    if (r >= rows || c >= cols)
        throw std::out_of_range("Matrix index out of range");

    return data[r * cols + c];
}

void Matrix::randomize() {
    static thread_local std::mt19937 generator(
        std::random_device{}()
    );

    double limit = std::sqrt(6.0 / (rows + cols));

    std::uniform_real_distribution<double> distribution(
        -limit, limit
    );

    for (double& value : data)
        value = distribution(generator);
}

void Matrix::fill(double value) {
    for (double& element : data)
        element = value;
}