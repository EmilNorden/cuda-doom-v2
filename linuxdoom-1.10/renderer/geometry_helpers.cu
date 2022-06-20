#include "geometry_helpers.cuh"
#include "device_random.cuh"
#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

namespace geom {


    __device__ glm::vec3 get_object_space_normal_from_normal_map(
            const glm::vec3 &object_space_normal,
            const glm::vec3 &sampled_normal,
            const glm::vec3 &object_space_tangent,
            const glm::vec3 &object_space_bitangent) {
        glm::mat3x3 tbn{object_space_tangent.x, object_space_tangent.y,
                        object_space_tangent.z, object_space_bitangent.x,
                        object_space_bitangent.y, object_space_bitangent.z,
                        object_space_normal.x, object_space_normal.y,
                        object_space_normal.z};

        return tbn * ((sampled_normal * 2.0f) - glm::vec3{1, 1, 1});
    }


    __device__ glm::vec3 random_unit_in_hemisphere(const glm::vec3 &normal, RandomGenerator &random) {
        float mt_x = (random.value()) - 0.5f;
        float mt_y = (random.value()) - 0.5f;
        float mt_z = (random.value()) - 0.5f;

        auto vector = glm::normalize(glm::vec3{mt_x, mt_y, mt_z});
        if (glm::dot(vector, normal) < 0)
            vector = glm::vec3(-vector.x, -vector.y, -vector.z);

        return vector;
    }

    __device__ glm::vec3
    random_unit_in_cone(const glm::vec3 &cone_direction, float cone_angle, RandomGenerator &random) {
        auto perpendicular_vector =
                glm::abs(glm::dot(cone_direction, {0, 1, 0})) > 0.95f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);

        auto cross_vector = glm::normalize(glm::cross(cone_direction, perpendicular_vector));

        auto s = random.value();
        auto r = random.value();

        auto h = glm::cos(cone_angle);

        auto phi = 2 * glm::pi<float>() * s;
        auto z = h + (1.0f - h) * r;
        auto sin_t = glm::sqrt(1 - z * z);
        auto x = glm::cos(phi) * sin_t;
        auto y = glm::sin(phi) * sin_t;

        return perpendicular_vector * x + cross_vector * y + cone_direction * z;
    }

}