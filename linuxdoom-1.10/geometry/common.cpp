#include "common.h"
#include "triangulation.h"

namespace geometry {

    bool is_point_on_left_side_of_line(const glm::vec2 &line_v1, const glm::vec2 &line_v2, const glm::vec2 &point) {
        Triangle t(line_v1, line_v2, point);

        return t.area() > 0.0f;
    }

}