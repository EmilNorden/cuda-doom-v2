//
// Created by emil on 2021-05-15.
//

#ifndef RENDERER_COORDINATES_H
#define RENDERER_COORDINATES_H

#include <glm/glm.hpp>

struct WorldSpaceCoordinate {
public:
    __device__ WorldSpaceCoordinate(glm::vec3 coordinate) : m_coordinate(coordinate) {}

    [[nodiscard]] __device__ const glm::vec3& as_vec3() const { return m_coordinate; }
private:
    glm::vec3 m_coordinate;
};

struct ObjectSpaceCoordinate {
public:
    __device__ ObjectSpaceCoordinate(glm::vec3 coordinate) : m_coordinate(coordinate) {}

    [[nodiscard]] __device__ const glm::vec3& as_vec3() const { return m_coordinate; }
private:
    glm::vec3 m_coordinate;
};

#endif //RENDERER_COORDINATES_H
