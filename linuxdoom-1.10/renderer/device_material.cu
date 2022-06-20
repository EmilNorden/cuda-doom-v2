#include "device_material.cuh"
#include "device_texture.cuh"

__device__ std::uint8_t DeviceMaterial::sample_diffuse(const glm::vec2 &uv) const
{
    return m_diffuse->sample(uv * m_uv_scale);
}
/*
__device__ glm::vec3 DeviceMaterial::sample_normal(const glm::vec2 &uv) const
{
    return m_normal_map->sample(uv * m_uv_scale);
}

__device__ glm::vec3 DeviceMaterial::sample_roughness(const glm::vec2 &uv) const
{
    return m_roughness_map->sample(uv * m_uv_scale);
}
 */