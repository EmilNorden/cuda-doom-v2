#ifndef DOOM_SQUARE_H_
#define DOOM_SQUARE_H_

#include <glm/glm.hpp>
#include "device_material.cuh"

class Ray;

class DeviceTexture;

struct Square {
    __device__ __host__ Square(const glm::vec3 &top_left, const glm::vec3 &horizontal_vec,
                               const glm::vec3 &vertical_vec, const glm::vec2 &uv_scale,
                               DeviceMaterial material)
            : top_left(top_left),
              horizontal_vec(glm::normalize(horizontal_vec)),
              vertical_vec(glm::normalize(vertical_vec)),
              horizontal_len(glm::length(horizontal_vec)),
              vertical_len(glm::length(vertical_vec)),
              uv_scale(uv_scale),
              uv_offset(0.0f),
              material(material),
              texture_wrapping(true),
              lower_unpegged(false) {
        auto N = glm::normalize(glm::cross(horizontal_vec, vertical_vec));
        normal.x = N.x;
        normal.y = N.z;
    }

    glm::vec3 top_left;
    glm::vec3 horizontal_vec;
    glm::vec3 vertical_vec;
    glm::vec2 normal; // Due to the nature of DOOM maps, it suffices to store wall normals in the XZ plane
    glm::vec2 uv_scale;
    float horizontal_len;
    float vertical_len;
    float uv_offset;
    bool texture_wrapping;
    bool lower_unpegged;

    DeviceMaterial material;
};

// TODO: Temporarily making this __host__. Not needed, remove later
__host__ __device__ bool
intersects_wall(const Ray &ray, Square *wall, float &hit_distance, float &u, float &v, glm::vec3 &out_normal);

#endif