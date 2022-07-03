#ifndef RENDERER_DEVICE_MATERIAL_CUH_
#define RENDERER_DEVICE_MATERIAL_CUH_

#include <glm/glm.hpp>

class DeviceTexture;

class DeviceMaterial {
public:
    __host__ __device__ DeviceMaterial() : m_diffuse(nullptr), m_uv_scale(1.0f) {}

    __host__ __device__ explicit DeviceMaterial(DeviceTexture *diffuse)
            : m_diffuse(diffuse), m_uv_scale(1.0f) {}

    __device__ __host__ const DeviceTexture *diffuse_map() const { return m_diffuse; }

    __device__ __host__ void set_diffuse_map(DeviceTexture *texture) { m_diffuse = texture; }

    [[nodiscard]] __device__ bool has_normal_map() const { return m_normal_map != nullptr; }

    [[nodiscard]] __device__ bool has_roughness_map() const { return m_roughness_map != nullptr; }

    __device__ __host__ void set_normal_map(DeviceTexture *texture) { m_normal_map = texture; }

    __device__ __host__ void set_roughness_map(DeviceTexture *texture) { m_roughness_map = texture; }

    [[nodiscard]] __device__ __host__ const glm::vec3 &emission() const { return m_emission; }

    __device__ __host__ void set_emission(const glm::vec3 &value) { m_emission = value; }

    [[nodiscard]] __device__ __host__ float reflectivity() const { return m_reflectivity; }

    __device__ __host__ void set_reflectivity(float value) { m_reflectivity = value; }

    [[nodiscard]] __device__ __host__ float translucence() const { return m_translucence; }

    __device__ __host__ void set_translucence(float value) { m_translucence = value; }

    [[nodiscard]] __device__ __host__ glm::vec2 uv_scale() const { return m_uv_scale; }

    __device__ __host__ void set_uv_scale(const glm::vec2 &value) { m_uv_scale = value; }

    [[nodiscard]]  __device__ std::uint16_t sample_diffuse(const glm::vec2 &uv) const;

    /*[[nodiscard]] __device__ glm::vec3 sample_normal(const glm::vec2 &uv) const;

    [[nodiscard]] __device__ glm::vec3 sample_roughness(const glm::vec2 &uv) const;*/

private:
    DeviceTexture *m_diffuse;
    DeviceTexture *m_normal_map{};
    DeviceTexture *m_roughness_map{}; // TODO: Implement support for 8bpp textures
    glm::vec3 m_emission{};
    float m_reflectivity{};
    float m_translucence{};
    glm::vec2 m_uv_scale;
};

#endif