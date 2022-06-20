#include "device_random.cuh"
#include "cuda_utils.cuh"

__global__ void init_random_states(curandState *states, size_t pool_size, unsigned long long seed) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < pool_size) {
        curand_init(seed, i, 0, &states[i]);
    }
}

void RandomGeneratorPool::init_global_state(size_t pool_size, unsigned long long seed) {
    dim3 block(256, 1 ,1);
    dim3 grid(pool_size / block.x, 1, 1);
    init_random_states<<<grid, block>>>(m_global_state, pool_size, seed);
    cuda_assert(cudaDeviceSynchronize());
}