#ifndef RENDERER_GEOMETRY_HELPERS_CUH_
#define RENDERER_GEOMETRY_HELPERS_CUH_

#include <glm/glm.hpp>

class RandomGenerator;

namespace geom{
    __device__ glm::vec3 get_object_space_normal_from_normal_map(
            const glm::vec3& object_space_normal,
            const glm::vec3& sampled_normal,
            const glm::vec3& object_space_tangent,
            const glm::vec3& object_space_bitangent);

    __device__ glm::vec3 random_unit_in_hemisphere(const glm::vec3& normal, RandomGenerator& random);

    __device__ glm::vec3 random_unit_in_cone(const glm::vec3& cone_direction, float cone_angle, RandomGenerator& random);
}

#endif