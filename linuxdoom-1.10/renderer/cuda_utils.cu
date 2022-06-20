//
// Created by emil on 2021-05-09.
//

#include "cuda_utils.cuh"
#include <iostream>

void cuda_assert(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cerr << "CUDA call failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}
