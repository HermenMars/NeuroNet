# C++ Neural Network

A simple feedforward neural network implemented from scratch in C++17.

## Features

- Matrix operations
- ReLU and Sigmoid activation functions
- Forward propagation
- Backpropagation
- Gradient descent
- Model serialization
- XOR training example

## Requirements

- GCC or Clang
- C++17
- CMake 3.20+

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```bash
./build/neural_network
```

## Architecture

The example uses:

2 input neurons -> 8 hidden neurons -> 1 output neuron.

## License

No Licence