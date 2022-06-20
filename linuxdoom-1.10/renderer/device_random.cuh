#ifndef RENDERER_DEVICE_RANDOM_CUH_
#define RENDERER_DEVICE_RANDOM_CUH_

#include <assert.h>
#include <curand_kernel.h>
#include <iostream>

class RandomGenerator {
public:
    __device__ explicit RandomGenerator(curandState *state)
            : m_state(state) {
    }

    __device__ float value() {
        return curand_uniform(m_state);
    }
private:
    curandState *m_state;
};

class RandomGeneratorPool {
public:
    __host__ RandomGeneratorPool(size_t pool_size, unsigned long long seed)
        : m_global_state(nullptr), m_pool_size(pool_size) {
        cudaMalloc(&m_global_state, sizeof(curandState) * pool_size);

        init_global_state(pool_size, seed);
    }

    __device__ RandomGenerator get_generator(size_t index) {
        assert(index < m_pool_size);
        return RandomGenerator{&m_global_state[index]};
    }
private:
    size_t m_pool_size;
    curandState* m_global_state;

    void init_global_state(size_t pool_size, unsigned long long seed);
};

RandomGeneratorPool* create_random_generator_pool(size_t pool_size, unsigned long long seed);

#endif