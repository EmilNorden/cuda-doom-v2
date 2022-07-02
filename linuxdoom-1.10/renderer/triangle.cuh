#ifndef DOOM_TRIANGLE_CUH_
#define DOOM_TRIANGLE_CUH_

#include <glm/glm.hpp>
#include <vector>
#include <optional>

class Ray;

class DeviceTexture;

struct Triangle {
    Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, DeviceTexture *texture)
            : v0(v0), v1(v1), v2(v2), texture(texture) {}
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;

    DeviceTexture *texture;
};

__device__ bool intersects_triangle(const Ray &ray, Triangle *triangle, float &hit_distance, float &u, float &v);

#endif