#ifndef RENDERER_DEVICE_FIXED_VECTOR_CUH_
#define RENDERER_DEVICE_FIXED_VECTOR_CUH_

template<typename T>
class DeviceFixedVector {
public:
    DeviceFixedVector() : m_items(nullptr), m_count(0), m_capacity(0) {}

    explicit DeviceFixedVector(size_t capacity) : m_items(nullptr), m_count(0), m_capacity(capacity) {
        cuda_assert(cudaMallocManaged(&m_items, sizeof(T) * m_capacity));
    }

    template<template<typename, typename> typename ContainerType, typename AllocatorType>
    explicit DeviceFixedVector(const ContainerType<T, AllocatorType> &items) : DeviceFixedVector(items, items.size()) {}

    template<template<typename, typename> typename ContainerType, typename AllocatorType>
    DeviceFixedVector(const ContainerType<T, AllocatorType> &items, size_t capacity) : m_items(nullptr),
                                                                                       m_count(items.size()),
                                                                                       m_capacity(capacity) {
        transfer_vector_to_device_memory(items, &m_items, capacity);
    }

    __device__ __host__ T operator[](int i) {
        return m_items[i];
    }

    void push_back(T item) {
        assert(m_count < m_capacity);
        m_items[m_count] = item;
        m_count++;
    }

    bool remove(T item) {
        for (int i = 0; i < m_count; ++i) {
            if (item == m_items[i]) {
                m_items[i] = m_items[m_count - 1];
                m_count--;
                return true;
            }
        }
        return false;
    }

    __host__ void reset() {
        if (m_items) {
            cudaFree(m_items);
        }
        m_items = nullptr;
        m_count = 0;
        m_capacity = 0;
    }

    template<template<typename, typename> typename ContainerType, typename AllocatorType>
    __host__ void reset(const ContainerType<T, AllocatorType> &items) {
        reset();
        transfer_vector_to_device_memory(items, &m_items, items.size());
        m_count = m_capacity = items.size();
    }

    __device__ __host__ T *data() const { return m_items; }


    [[nodiscard]] __device__ __host__  size_t count() const { return m_count; }

    [[nodiscard]] __device__ __host__  size_t capacity() const { return m_capacity; }

    [[nodiscard]] __device__ __host__  bool is_full() const { return m_count >= m_capacity; }

private:
    T *m_items;
    size_t m_count;
    size_t m_capacity;
};

#endif