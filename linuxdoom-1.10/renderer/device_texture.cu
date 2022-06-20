#include "device_texture.cuh"
#include <algorithm>
#include "cuda_utils.cuh"

DeviceTexture::DeviceTexture(const std::vector<uint16_t> &pixels, size_t width, size_t height)
        : m_data(nullptr), m_width(width), m_height(height) {

    transfer_vector_to_device_memory(pixels, &m_data);
}

__device__ std::uint16_t get_color_at(std::uint16_t *data, size_t x, size_t y, size_t width, size_t height) {
    x = x % width;
    y = y % height;
    auto index = (y * width) + x;
    return data[index];
}

__device__ std::uint16_t DeviceTexture::sample(const glm::vec2 &uv) const {
    auto x = static_cast<size_t>(uv.x * (m_width - 1));
    auto y = static_cast<size_t>(uv.y * (m_height - 1));

    return get_color_at(m_data, x, y, m_width, m_height);
}