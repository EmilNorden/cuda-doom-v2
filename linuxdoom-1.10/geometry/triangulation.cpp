#include "triangulation.h"
#include "polygon.h"
#include "common.h"

namespace geometry {

    Triangle::Triangle(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2)
            : m_v0(v0), m_v1(v1), m_v2(v2) {
    }

    /* Taken from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle*/
    bool Triangle::contains(const glm::vec2 &vertex) const {
        auto sign = [](const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2) {
            return (v0.x - v2.x) * (v1.y - v2.y) - (v1.x - v2.x) * (v0.y - v2.y);
        };

        auto d1 = sign(vertex, m_v0, m_v1);
        auto d2 = sign(vertex, m_v1, m_v2);
        auto d3 = sign(vertex, m_v2, m_v0);

        auto has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        auto has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

        return !(has_neg && has_pos);
    }

    float Triangle::area() const {
        return (m_v0.x * (m_v2.y - m_v1.y)) + (m_v1.x * (m_v0.y - m_v2.y)) + (m_v2.x * (m_v1.y - m_v0.y));
    }

    [[nodiscard]] Winding Triangle::winding() const {
        return is_point_on_left_side_of_line(m_v0, m_v2, m_v1) ? Winding::Clockwise : Winding::CounterClockwise;
    }

    std::vector<Triangle> triangulate(const Polygon &polygon) {
        std::vector<Triangle> triangles;
        if (polygon.size() == 3) {
            triangles.emplace_back(polygon[0], polygon[1], polygon[2]);
        }

        auto polygon_size = polygon.size();
        std::vector<bool> clipped_vertices(polygon_size, false);

        while (polygon_size > 2) {
            // FIND EAR
            auto any_clipped = false;
            for (int i = 0; i < polygon.size(); ++i) {
                if (clipped_vertices[i]) {
                    continue;
                }

                auto previous_index = i == 0 ? polygon.size() - 1 : i - 1;
                while (clipped_vertices[previous_index]) {
                    previous_index = previous_index == 0 ? polygon.size() - 1 : previous_index - 1;
                }

                auto next_index = (i + 1) % polygon.size();
                while (clipped_vertices[next_index]) {
                    next_index = (next_index + 1) % polygon.size();
                }

                auto v0 = polygon.vertices()[previous_index];
                auto v1 = polygon.vertices()[i];
                auto v2 = polygon.vertices()[next_index];

                Triangle triangle(v0, v1, v2);

                if (triangle.area() == 0.0f) {
                    // Area is 0. All vertices must be colinear.
                    continue;
                }

                if (triangle.winding() == Winding::Clockwise) {
                    // Assuming CCW  winding, the point should be on the right side.
                    // Move on to the next vertex in the polygon
                    continue;
                }

                bool contains_other_vertices = false;
                for (auto vertex: polygon.vertices()) {
                    if (vertex != v0 && vertex != v1 && vertex != v2 && triangle.contains(vertex)) {
                        contains_other_vertices = true;
                        break;
                    }
                }
                if(contains_other_vertices) {
                    continue;
                }

                triangles.push_back(triangle);
                any_clipped = true;
                clipped_vertices[i] = true;
                polygon_size -= 1;
                // printf("polygon size is now %lu, i is %d\n", polygon_size, i);
                if (polygon_size < 3) {
                    break;
                }
            }

            if (!any_clipped) {
                printf("Cant clip anymore :( ABorting\n");
                break;
            }
        }

        return triangles;
    }

    float triangle_area_2d(const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3) {
        return (v1.x * (v3.y - v2.y)) + (v2.x * (v1.y - v3.y)) + (v3.x * (v2.y - v1.y));
    }

    bool is_triangle_cw_winding(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2) {
        return is_point_on_left_side_of_line(v0, v2, v1);
    }

    /* Taken from https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle*/
    bool
    is_point_in_triangle_2d(const glm::vec2 &point, const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2) {
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
}