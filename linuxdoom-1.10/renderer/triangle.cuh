#ifndef DOOM_TRIANGLE_CUH_
#define DOOM_TRIANGLE_CUH_

#include <glm/glm.hpp>
#include <vector>
#include <optional>
#include "device_material.cuh"

class Ray;

class DeviceTexture;

struct Triangle {
    Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, DeviceMaterial material)
            : v0(v0), v1(v1), v2(v2), material(material) {}
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;

    DeviceMaterial material;
};

__device__ bool intersects_triangle(const Ray &ray, Triangle *triangle, float &hit_distance, float &u, float &v);

#endif