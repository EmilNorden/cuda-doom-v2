cmake_minimum_required (VERSION 3.19)
project (doom CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_subdirectory(linuxdoom-1.10)
add_subdirectory(test)