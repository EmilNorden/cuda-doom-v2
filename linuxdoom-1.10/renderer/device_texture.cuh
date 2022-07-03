#ifndef RENDERER_DEVICE_TEXTURE_CUH_
#define RENDERER_DEVICE_TEXTURE_CUH_

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

class DeviceTexture {
public:
    __host__ DeviceTexture(const std::vector<uint16_t> &pixels, size_t width, size_t height);

    [[nodiscard]] __device__ std::uint16_t sample(const glm::vec2 &uv) const;

    __host__ __device__ size_t width() const { return m_width; }
    __host__ __device__ size_t height() const {return m_height; }

    __host__ const std::uint16_t *data() const { return m_data; }

private:
    std::uint16_t *m_data;
    size_t m_width;
    size_t m_height;
};

#endif