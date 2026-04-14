cmake_minimum_required(VERSION 3.18)
project(Infera LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

set(SOURCES
    src/arena.cpp
    src/tensor.cpp
    src/ops.cu
    src/main.cpp
)

add_executable(infera ${SOURCES})
target_include_directories(infera PRIVATE include)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(infera PRIVATE CUDA::cudart)

# enable optimizations for benchmarking later
target_compile_options(infera PRIVATE -O3 -march=native)
