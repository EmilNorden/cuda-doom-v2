#ifndef DOOM_TRIANGULATION_H_
#define DOOM_TRIANGULATION_H_

#include "winding.h"
#include <vector>
#include <glm/glm.hpp>

namespace geometry {
    class Polygon;

    class Triangle {
    public:
        Triangle(const glm::vec2 &v0, const glm::vec2 &v1, const glm::vec2 &v2);

        [[nodiscard]] bool contains(const glm::vec2 &vertex) const;
        [[nodiscard]] float area() const;
        [[nodiscard]] Winding winding() const;
        [[nodiscard]] glm::vec2 v0() { return m_v0; }
        [[nodiscard]] glm::vec2 v1() { return m_v1; }
        [[nodiscard]] glm::vec2 v2() { return m_v2; }
    private:
        glm::vec2 m_v0;
        glm::vec2 m_v1;
        glm::vec2 m_v2;
    };

    std::vector<Triangle> triangulate(const Polygon &polygon);

    float triangle_area_2d(const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3);
}

#endif