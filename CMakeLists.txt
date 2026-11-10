cmake_minimum_required(VERSION 3.20)

project(NeuralNetwork
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(neural_network
    src/main.cpp
    src/NeuralNetwork.cpp
    src/Matrix.cpp
)

target_include_directories(neural_network PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(neural_network PRIVATE
    -Wall
    -Wextra
    -O3
)