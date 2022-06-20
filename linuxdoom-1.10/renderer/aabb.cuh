#ifndef RENDERER_AABB_CUH_
#define RENDERER_AABB_CUH_

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <iostream>

class AABB {
public:
    __host__ __device__ AABB(const glm::vec3& min, const glm::vec3& max)
        : m_min(min), m_max(max) {

    }

    __host__ AABB(const std::vector<glm::vec3>& points, float padding = 0.0f) {
        if(points.empty()) {
            throw std::runtime_error{"Cannot create AABB from empty vector"};
        }

        m_min = points[0];
        m_max = points[0];

        for(auto& point : points) {
            m_min = glm::min(m_min, point);
            m_max = glm::max(m_max, point);
        }

        m_min -= glm::vec3{padding};
        m_max += glm::vec3{padding};
    }

    __host__ __device__ const glm::vec3& min() const { return m_min; }
    __host__ __device__ const glm::vec3& max() const { return m_max; }

    friend std::ostream &operator<<(std::ostream &os, const AABB &bounds) {
        return os << "[" << bounds.min().x << ", " << bounds.min().y << ", " << bounds.min().z << "] - [" << bounds.max().x << ", " << bounds.max().y << ", " << bounds.max().z << "]";
    }

    
private:
    glm::vec3 m_min;
    glm::vec3 m_max;
};

#endif