#ifndef RENDERER_RAY_H_
#define RENDERER_RAY_H_

#include <glm/glm.hpp>
#include "coordinates.cuh"
#include "transform.cuh"

class Ray {
public:
    // TODO: Temporarily making this __host__. Not needed, remove later
    __host__ __device__ Ray(const glm::vec3& origin, const glm::vec3& direction, float refractive_index = 1.0f)
            : m_origin(origin), m_direction(direction), m_inverse_direction({1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z}), m_refractive_index(refractive_index) {

    }

    // TODO: Temporarily making this __host__. Not needed, remove later
    [[nodiscard]] __device__ __host__  const glm::vec3& origin() const { return m_origin; }
    // TODO: Temporarily making this __host__. Not needed, remove later
    [[nodiscard]] __device__ __host__ const glm::vec3& direction() const { return m_direction; }
    [[nodiscard]] __device__ const glm::vec3& inverse_direction() const { return m_inverse_direction; }
    [[nodiscard]] __device__ const float refractive_index() const { return m_refractive_index; }
protected:
    glm::vec3 m_origin;
    glm::vec3 m_direction;
    glm::vec3 m_inverse_direction;
    float m_refractive_index;
};


#endif