#include "triangle.cuh"
#include "ray.cuh"

#define EPSILON 9.99999997475243E-07

__device__ bool intersects_triangle(const Ray &ray, Triangle &triangle, float &hit_distance, float &u, float &v) {
    auto v1 = triangle.v0;

    /*
     *             glm::vec3 e1 = v2 - v1;
            glm::vec3 e2 = v3 - v1;
     * */
    glm::vec3 e1 = triangle.v1 - triangle.v0;
    glm::vec3 e2 = triangle.v2 - triangle.v0;

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

std::vector<glm::vec2> project_into_2d(const std::vector<glm::vec3> &polygon) {
    auto surface_normal = glm::vec3(0, 1, 0);

    auto project_axis_a = 0;
    auto project_axis_b = 1;
    auto inv = surface_normal.z;

    if (surface_normal.x > surface_normal.y) {
        if (surface_normal.x > surface_normal.z) {
            project_axis_a = 1;
            project_axis_b = 2;
            inv = surface_normal.x;
        }
    } else if (surface_normal.y > surface_normal.z) {
        project_axis_a = 0;
        project_axis_b = 2;
        inv = surface_normal.y;
    }

    if (inv < 0.0) {
        std::swap(project_axis_a, project_axis_b);
    }

    std::vector<glm::vec2> vertices;
    for (auto vertex: polygon) {
        vertices.emplace_back(
                vertex[project_axis_a],
                vertex[project_axis_b]
        );
    }

    return vertices;
}

float triangle_area_2d(const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3) {
    return (v1.x * (v3.y - v2.y)) + (v2.x * (v1.y - v3.y)) + (v3.x * (v2.y - v1.y));
}

bool is_point_on_left_side_of_line(const glm::vec2 &line_v1, const glm::vec2 &line_v2, const glm::vec2 &point) {
    return triangle_area_2d(line_v1, point, line_v2) > 0.0f;
}

bool is_triangle_cw_winding(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2){
    return is_point_on_left_side_of_line(v0, v2, v1);
}

/* Taken from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle*/
bool is_point_in_triangle_2d(const glm::vec2 &point, const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2) {
    auto sign = [](const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2) {
        return (v0.x - v2.x) * (v1.y - v2.y) - (v1.x - v2.x) * (v0.y - v2.y);
    };

    auto d1 = sign(point, v0, v1);
    auto d2 = sign(point, v1, v2);
    auto d3 = sign(point, v2, v0);

    auto has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    auto has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    return !(has_neg && has_pos);
}

bool tri_contains_other_verts_2d(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2,
                                 const std::vector<glm::vec2> &vertices) {
    for (auto vertex: vertices) {
        if (vertex != v0 && vertex != v1 && vertex != v2 && is_point_in_triangle_2d(vertex, v0, v1, v2)) {
            return true;
        }
    }

    return false;
}

std::vector<Triangle> triangulate_polygon(const std::vector<glm::vec3> &polygon, DeviceTexture* texture) {
    std::vector<Triangle> triangles;
    if (polygon.size() == 3) {
        triangles.push_back(Triangle{polygon[0], polygon[1], polygon[2], texture});
    }

    auto polygon_size = polygon.size();
    std::vector<bool> clipped_vertices(polygon_size, false);

    auto plane_vertices = project_into_2d(polygon);

    while (polygon_size > 2) { // Why did I do polygon_size > 3 before? That would skip the last triangle...
        // FIND EAR
        auto any_clipped = false;
        for (int i = 0; i < polygon.size(); ++i) {
            if (clipped_vertices[i]) {
                continue;
            }

            auto previous_index = i == 0 ? polygon.size() - 1 : i - 1;
            while (clipped_vertices[previous_index]) {
                previous_index = previous_index == 0 ? polygon.size()  - 1 : previous_index - 1;
            }

            auto next_index = (i + 1) % polygon.size();
            while (clipped_vertices[next_index]) {
                next_index = (next_index + 1) % polygon.size();
            }

            auto v0 = plane_vertices[previous_index];
            auto v1 = plane_vertices[i];
            auto v2 = plane_vertices[next_index];

            if(triangle_area_2d(v0, v1, v2) == 0.0f) {
                // Area is 0. All vertices must be colinear.
                continue;
            }

            if (is_triangle_cw_winding(v0, v2, v1)) {
                // Assuming CCW  winding, the point should be on the right side.
                // Move on to the next vertex in the polygon
                continue;
            }

            if (tri_contains_other_verts_2d(v0, v1, v2, plane_vertices)) {
                continue;
            }

            triangles.push_back(Triangle{polygon[previous_index], polygon[i], polygon[next_index], texture});
            any_clipped = true;
            clipped_vertices[i] = true;
            polygon_size -= 1;
            printf("polygon size is now %lu, i is %d\n", polygon_size, i);
            if (polygon_size < 3) {
                break;
            }
        }

        if(!any_clipped) {
            printf("Cant clip anymore :( ABorting\n");
            break;
        }
    }

    return triangles;
}