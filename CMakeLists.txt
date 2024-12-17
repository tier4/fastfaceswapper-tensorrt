cmake_minimum_required(VERSION 3.0)
project(fastfaceswapper-tensorrt VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wall -Wextra -Wpedantic -Wno-long-long")

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(CUDA REQUIRED)
find_package(absl REQUIRED)

set(CMAKE_CUDA_FLAG "${CMAKE_CUDA_FLAG} -O3")

include_directories(/usr/local/cuda/include include)
link_directories(/usr/local/cuda/lib64)

add_subdirectory(tools)
