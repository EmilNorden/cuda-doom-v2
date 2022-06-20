#ifndef RENDERER_DEVICE_STACK_CUH_
#define RENDERER_DEVICE_STACK_CUH_

#include <cstddef>
#include <assert.h>

template<size_t Capacity, typename T>
class DeviceStack {
public:
    __device__ __host__ explicit DeviceStack()
            : m_size(0) {
    }


    DeviceStack(const DeviceStack&) = delete;

    __device__ __host__  void push(const T &item) {
        assert(m_size < Capacity);
        m_data[m_size++] = item;
    }

    __device__ __host__  T pop() {
        assert(m_size > 0);
        return m_data[--m_size];
    }

    [[nodiscard]] __device__ bool is_empty() const {
        return m_size == 0;
    }

private:
    T m_data[Capacity];
    size_t m_size;
};

#endif