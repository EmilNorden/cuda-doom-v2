//
// Created by emil on 2022-07-02.
//

#include "polygon.h"

#include <utility>
#include <algorithm>
#include <FreeImage.h>
#include <fmt/core.h>

namespace geometry {
    bool on_segment(glm::vec2 p, glm::vec2 q, glm::vec2 r);

    int orientation(glm::vec2 p, glm::vec2 q, glm::vec2 r);

    bool lines_intersect(glm::vec2 p1, glm::vec2 q1, glm::vec2 p2, glm::vec2 q2);

    Winding calculate_polygon_winding(const std::vector<glm::vec2> &polygon) {

        float total_sum = 0.0f;
        for (int i = 0; i < polygon.size(); ++i) {
            auto next_index = i + 1;
            if (i == polygon.size() - 1) {
                next_index = 0;
            }

            total_sum += (polygon[next_index].x - polygon[i].x) * (polygon[next_index].y + polygon[i].y);
        }

        return total_sum >= 0.0f ? Winding::Clockwise : Winding::CounterClockwise;
    }

    Polygon::Polygon(std::vector<glm::vec2> vertices)
            : m_vertices(std::move(vertices)), m_weld_points(m_vertices.size(), false),
              m_winding(calculate_polygon_winding(m_vertices)) {

    }

    void Polygon::flip_winding() {
        std::reverse(m_vertices.begin(), m_vertices.end());
        std::reverse(m_weld_points.begin(), m_weld_points.end());

        if (m_winding == Winding::CounterClockwise) {
            m_winding = Winding::Clockwise;
        } else {
            m_winding = Winding::CounterClockwise;
        }
    }

    bool Polygon::contains_vertex(const glm::vec2 &vertex) const {
// Test all edges of the (possibly) containing polygon against an arbitrary line starting from the given point
        // NOTE: I am doing the check twice, with two different "abritrary lines". This is to avoid false positives when the lines become collinear with lines in the polygon.
        auto end_point1 = vertex + glm::vec2{0, -2000};
        auto end_point2 = vertex + glm::vec2{0, 2000};

        int intersections1 = 0;
        int intersections2 = 0;
        for (int i = 0; i < this->size(); ++i) {
            auto j = (i == this->size() - 1) ? 0 : i + 1;

            auto edge_start = m_vertices[i];
            auto edge_end = m_vertices[j];

            if (lines_intersect(vertex, end_point1, edge_start, edge_end)) {
                intersections1++;
            }

            if (lines_intersect(vertex, end_point2, edge_start, edge_end)) {
                intersections2++;
            }
        }

        return (intersections1 & 1) > 0 && (intersections2 & 1) > 0;
    }

    bool Polygon::intersects_polygon(const Polygon &other) const {
        /*for (auto &p: other.vertices()) {
            if (contains_vertex(p)) {
                return true;
            }
        }

         return false;*/

        return std::any_of(other.vertices().begin(), other.vertices().end(), [&](auto p) {
            return contains_vertex(p);
        });
    }

    void Polygon::combine_with(const Polygon &other) { // TODO: Add weld_points as a field of Polygon
        auto found_vertex_pair = false;
        int parent_vertex = -1;
        int child_vertex = -1;
        float best_distance = FLT_MAX;
        for (int i = 0; i < other.size(); ++i) {
            for (auto j = 0; j < size(); ++j) {
                auto distance = glm::length(other[i] - m_vertices[j]);
                auto is_already_weld_point = m_weld_points[j];
                if (is_already_weld_point) {
                    distance *= 1000; // Dont diregard the point entirely, but make it less likely to be choosen.
                }
                if (distance < best_distance) {
                    best_distance = distance;
                    parent_vertex = j;
                    child_vertex = i;
                }
            }
        }

        if (parent_vertex == -1) {
            printf("ABORTING. Cant find best parent/child vertex for polygon combination\n");
            exit(-1);
        }

        m_weld_points[parent_vertex] = true;
        auto cv = other[child_vertex];

        std::vector<glm::vec2> child_vertices(other.vertices().begin(), other.vertices().end());
        std::rotate(child_vertices.begin(), child_vertices.begin() + child_vertex, child_vertices.end());

        child_vertices.push_back(cv);
        child_vertices.push_back(m_vertices[parent_vertex]);

        m_vertices.insert(m_vertices.begin() + parent_vertex + 1, child_vertices.begin(),
                          child_vertices.end());

        std::vector<bool> new_points(child_vertices.size(), false);
        m_weld_points.insert(m_weld_points.begin() + parent_vertex + 1, new_points.begin(), new_points.end());
        m_weld_points[parent_vertex + 1] = true;
        m_weld_points[parent_vertex + child_vertices.size()] = true;
        m_weld_points[parent_vertex + child_vertices.size() - 1] = true;
    }

    void Polygon::assert_winding(Winding winding) {
        if (m_winding == winding) {
            return;
        }

        flip_winding();
    }

    void Polygon::debug_image(const std::string &name) const {
        constexpr size_t image_width = 1024;
        constexpr size_t image_height = 1024;
        auto image = FreeImage_Allocate(1024, 1024, 32);

        for (int y = 0; y < image_height; ++y) {
            for (int x = 0; x < image_width; ++x) {
                RGBQUAD rgb;
                rgb.rgbBlue = rgb.rgbGreen = rgb.rgbRed = rgb.rgbReserved = 255;

                FreeImage_SetPixelColor(image, x, y, &rgb);
            }
        }

        auto smallest = glm::vec2(FLT_MAX, FLT_MAX);
        auto largest = glm::vec2(FLT_MIN, FLT_MIN);

        for (auto vertex: m_vertices) {
            smallest = glm::min(smallest, vertex);
            largest = glm::max(largest, vertex);
        }

        smallest = smallest - glm::vec2(10.0, 10.0);
        largest = largest + glm::vec2(10.0, 10.0);

        auto polygon_size = largest - smallest;

        auto get_slope = [](glm::vec2 &start, glm::vec2 &end) -> std::optional<float> {
            if (start.x == end.x) {
                return std::nullopt;
            }

            auto slope = (end.y - start.y) / (end.x - start.x);
            if (glm::abs(slope) > 100000.0f) {
                // slope is steep enough to handle as vertical
                return std::nullopt;
            }

            return slope;
        };

        auto get_intercept = [](glm::vec2 &start, std::optional<float> &slope) -> float {
            if (slope.has_value()) {
                return start.y - slope.value() * start.x;
            }
            return start.x;
        };

        for (int i = 0; i < size(); ++i) {
            auto next_index = (i + 1) % size();
            auto from_vertex = m_vertices[i];
            auto to_vertex = m_vertices[next_index];

            auto start = glm::vec2(
                    ((from_vertex.x - smallest.x) / polygon_size.x) * (image_width - 1.0),
                    ((from_vertex.y - smallest.y) / polygon_size.y) * (image_height - 1.0)
            );

            auto end = glm::vec2(
                    ((to_vertex.x - smallest.x) / polygon_size.x) * (image_width - 1.0),
                    ((to_vertex.y - smallest.y) / polygon_size.y) * (image_height - 1.0)
            );

            auto slope = get_slope(start, end);
            auto intercept = get_intercept(start, slope);

            auto previous_distance = FLT_MAX;

            auto current_pos = glm::vec2(start.x, start.y);

            while (glm::length(current_pos - end) < previous_distance) {
                previous_distance = glm::length(current_pos - end);

                auto base_increment = 0.1f;
                if (slope.has_value()) {
                    auto step_increment = start.x > end.x ? -base_increment : base_increment;
                    auto diff = end.x - start.x;
                    if (glm::abs(diff) < glm::abs(step_increment)) {
                        step_increment = diff;
                    }
                    current_pos.x += step_increment;
                    current_pos.y = slope.value() * current_pos.x + intercept;
                } else {
                    auto step_increment = start.y > end.y ? -base_increment : base_increment;
                    auto diff = end.y - start.y;
                    if (glm::abs(diff) < glm::abs(step_increment)) {
                        step_increment = diff;
                    }
                    current_pos.y += step_increment;
                }

                RGBQUAD rgb;
                rgb.rgbRed = rgb.rgbGreen = rgb.rgbBlue = 0;
                rgb.rgbReserved = 255;
                FreeImage_SetPixelColor(image, (int) current_pos.x, (int) current_pos.y, &rgb);
            }
        }

        FreeImage_Save(FIF_PNG, image, fmt::format("/home/emil/doom_wads/sectors/polygon_{}.png", name).c_str());
    }

// Given three collinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
    bool on_segment(glm::vec2 p, glm::vec2 q, glm::vec2 r) {
        if (q.x <= glm::max(p.x, r.x) && q.x >= glm::min(p.x, r.x) &&
            q.y <= glm::max(p.y, r.y) && q.y >= glm::min(p.y, r.y))
            return true;

        return false;
    }

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
    int orientation(glm::vec2 p, glm::vec2 q, glm::vec2 r) {
        // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
        // for details of below formula.
        int val = (q.y - p.y) * (r.x - q.x) -
                  (q.x - p.x) * (r.y - q.y);

        if (val == 0) return 0;  // collinear

        return (val > 0) ? 1 : 2; // clock or counterclock wise
    }

// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
    bool lines_intersect(glm::vec2 p1, glm::vec2 q1, glm::vec2 p2, glm::vec2 q2) {
        // Find the four orientations needed for general and
        // special cases
        int o1 = orientation(p1, q1, p2);
        int o2 = orientation(p1, q1, q2);
        int o3 = orientation(p2, q2, p1);
        int o4 = orientation(p2, q2, q1);

        // General case
        if (o1 != o2 && o3 != o4)
            return true;

        // Special Cases
        // p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if (o1 == 0 && on_segment(p1, p2, q1)) return true;

        // p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if (o2 == 0 && on_segment(p1, q2, q1)) return true;

        // p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if (o3 == 0 && on_segment(p2, p1, q2)) return true;

        // p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if (o4 == 0 && on_segment(p2, q1, q2)) return true;

        return false; // Doesn't fall in any of the above cases
    }
}
