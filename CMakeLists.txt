cmake_minimum_required(VERSION 3.20)
project(ssl_smoother_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake at your venv so it finds the right Python
set(Python3_ROOT_DIR "/Users/bthomp23/PycharmProjects/differential_dynamics/.venv")
set(Python3_FIND_STRATEGY LOCATION)

find_package(Python3 COMPONENTS Development REQUIRED)

# (Optional) also query torch’s include paths as you already do
set(PYTHON_EXECUTABLE "/Users/bthomp23/PycharmProjects/differential_dynamics/.venv/bin/python")
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c
        "from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
        OUTPUT_VARIABLE TORCH_INCLUDES
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(TORCH_INCLUDES_LIST NATIVE_COMMAND "${TORCH_INCLUDES}")

add_library(ssl_smoother_cpu_dbg MODULE csrc/ssl_smoother.cpp)

# Make Python C API visible (fixes Py_GetVersion/Python.h)
target_include_directories(ssl_smoother_cpu_dbg PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(ssl_smoother_cpu_dbg PRIVATE Python3::Python)

# Torch/pybind11 headers
target_include_directories(ssl_smoother_cpu_dbg PRIVATE ${TORCH_INCLUDES_LIST})

# Define module name for PYBIND11_MODULE
target_compile_definitions(ssl_smoother_cpu_dbg PRIVATE
        TORCH_EXTENSION_NAME=ssl_smoother_cpu_dbg
        PY_SSIZE_T_CLEAN           # recommended for Python.h
)

# Debug-friendly flags for IDE
target_compile_options(ssl_smoother_cpu_dbg PRIVATE -g -O0 -fno-omit-frame-pointer)