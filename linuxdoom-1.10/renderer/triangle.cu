#include "triangle.cuh"
#include "ray.cuh"
#include "cuda_utils.cuh"

#define EPSILON 9.99999997475243E-07

__device__ bool intersects_triangle(const Ray &ray, Triangle *triangle, float &hit_distance, float &u, float &v) {
    auto v1 = triangle->v0;

    /*
     *             glm::vec3 e1 = v2 - v1;
            glm::vec3 e2 = v3 - v1;
     * */
    glm::vec3 e1 = triangle->v1 - triangle->v0;
    glm::vec3 e2 = triangle->v2 - triangle->v0;

    // Begin calculating determinant - also used to calculate u parameter
    glm::vec3 P = glm::cross(ray.direction(), e2); // m_direction.cross(e2);
    // if determinant is near zero, ray lies in plane of triangle

    float det = glm::dot(e1, P); // e1.dot(P);

    /*if (det > -EPSILON && det < EPSILON)
        return false;*/

    // BACK-FACE CULLING

    /*if (det < 0.00000001f) {
        return false;
    }*/
    /*if (!inside_geometry && det < EPSILON) {
        return false;
    }*/

    float inv_det = 1.0f / det;

    // calculate distance from V1 to ray origin
    glm::vec3 T = ray.origin() - v1;

    // Calculate u parameter and test bound
    u = glm::dot(T, P) * inv_det;
    // The intersection lies outside of the triangle
    if (u < 0.f || u > 1.f)
        return false;

    // Prepare to test v parameter
    glm::vec3 Q = glm::cross(T, e1); // T.cross(e1);

    // Calculate V parameter and test bound
    v = glm::dot(ray.direction(), Q) * inv_det;
    // The intersection lies outside of the triangle
    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = glm::dot(e2, Q) * inv_det;

    if (t > EPSILON) { // ray intersection
        hit_distance = t;
        auto hit_pos = ray.origin() + (ray.direction() * t);
        u = fabs(hit_pos.x) / 63.0f;
        v = fabs(hit_pos.z) / 63.0f;
        return true;
    }

    // No hit, no win
    return false;
}