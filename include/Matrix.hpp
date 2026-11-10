#pragma once

#include <vector>
#include <cstddef>

class Matrix {
public:
    std::size_t rows;
    std::size_t cols;
    std::vector<double> data;

    Matrix(std::size_t rows, std::size_t cols);

    double& operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;

    void randomize();
    void fill(double value);
};