#include "square.cuh"
#include "ray.cuh"

// TODO: Temporarily making this __host__. Not needed, remove later
__host__ __device__ bool intersects_wall(const Ray &ray, Square* wall, float &hit_distance, float &u, float &v, glm::vec3 &out_normal) {
    auto N = glm::normalize(glm::cross(wall->vertical_vec, wall->horizontal_vec));


    auto dir_normal_dot_product = glm::dot(ray.direction(), N);
    // is this check needed?
    /*if(dir_normal_dot_product >= 0.0f) {
        return false;
    }*/

    auto a = glm::dot(wall->top_left - ray.origin(), N) / dir_normal_dot_product;

    if (a < 0.0f) {
        return false;
    }

    auto P = ray.origin() + (a * ray.direction());

    auto P0P = P - wall->top_left;

    auto Q1 = glm::dot(P0P, wall->horizontal_vec);
    if(Q1 < 0.0 || Q1 > wall->horizontal_len) {
        return false;
    }

    auto Q2 = glm::dot(P0P, wall->vertical_vec);
    if(Q2 < 0.0 || Q2 > wall->vertical_len) {
        return false;
    }

    //u = Q1 * 0.01; // wall.uv_scale.x;
    //v = Q2 * 0.01; // wall.uv_scale.y;

    u = Q1 * wall->uv_scale.x;
    v = Q2 * wall->uv_scale.y;

    hit_distance = a;
    out_normal = N;
    return true;
}