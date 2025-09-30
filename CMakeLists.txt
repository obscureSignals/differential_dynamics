cmake_minimum_required(VERSION 3.29)
project(differential_dynamics)

set(CMAKE_CXX_STANDARD 14)

add_executable(differential_dynamics
    csrc/sigmoid_smoother.cpp
    csrc/ssl_smoother.cpp)
