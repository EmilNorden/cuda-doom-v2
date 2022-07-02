#ifndef DOOM_GEOMETRY_COMMON_H_
#define DOOM_GEOMETRY_COMMON_H_

#include <glm/glm.hpp>

namespace geometry {
    bool is_point_on_left_side_of_line(const glm::vec2 &line_v1, const glm::vec2 &line_v2, const glm::vec2 &point);
}

#endif