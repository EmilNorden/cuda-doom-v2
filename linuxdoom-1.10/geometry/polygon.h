#ifndef DOOM_POLYGON_H
#define DOOM_POLYGON_H

#include <vector>
#include <glm/glm.hpp>

enum class Winding {
    Clockwise, CounterClockwise
};

class Polygon {
public:
    explicit Polygon(std::vector<glm::vec2> vertices);

    [[nodiscard]] const std::vector<glm::vec2> &vertices() const { return m_vertices; }

    [[nodiscard]] size_t size() const { return m_vertices.size(); }

    [[nodiscard]] Winding winding() const { return m_winding; }

    [[nodiscard]] bool contains_vertex(const glm::vec2 &vertex) const;

    [[nodiscard]] bool intersects_polygon(const Polygon& other) const;

    void combine_with(const Polygon& other, std::vector<bool> &weld_points);

    void assert_winding(Winding winding);

    glm::vec2 operator[](int i) const {
        return m_vertices[i];
    }

    void flip_winding();

private:
    std::vector<glm::vec2> m_vertices;
    Winding m_winding;
};


#endif //DOOM_POLYGON_H
