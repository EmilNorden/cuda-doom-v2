#ifndef RENDERER_TRANSFORM_CUH_
#define RENDERER_TRANSFORM_CUH_

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <optional>

class Transform {
public:
    explicit Transform(const glm::mat4x4 &transform)
            : m_transform(transform) {

    }

    [[nodiscard]] __device__ glm::vec3 transform_coordinate(const glm::vec3 &coordinate) const {
        return glm::vec3(m_transform * glm::vec4(coordinate, 1));
    }

    [[nodiscard]] __device__ glm::vec3 transform_vector(const glm::vec3 &vector) const {
        return glm::normalize(glm::vec3(m_transform * glm::vec4(vector, 0)));
    }

protected:
    glm::mat4x4 m_transform;
};

class TransposeInverseWorldTransform : public Transform {
public:
    explicit TransposeInverseWorldTransform(const glm::mat4x4 &mat)
            : Transform(mat) {
    }

    [[nodiscard]] __device__ glm::vec3 transform_normal(const glm::vec3 &normal) const {
        return glm::normalize(glm::vec3(m_transform * glm::vec4(normal, 0)));
    }
};

class InverseWorldTransform : public Transform {
public:
    explicit InverseWorldTransform(const glm::mat4x4 &mat)
            : Transform(mat) {
    }

    [[nodiscard]] TransposeInverseWorldTransform transpose() const {
        return TransposeInverseWorldTransform{glm::transpose(m_transform)};
    }
};

class WorldTransform : public Transform {
public:
    explicit WorldTransform(const glm::mat4x4 &mat = glm::mat4(1.0))
            : Transform(mat) {
    }

    [[nodiscard]] InverseWorldTransform invert() const {
        return InverseWorldTransform{glm::inverse(m_transform)};
    }
};

class WorldTransformBuilder {
public:
    WorldTransformBuilder &with_object_space_translation(const glm::vec3 &translation) {
        m_object_space_translation = translation;
        return *this;
    }

    WorldTransformBuilder &with_translation(const glm::vec3 &translation) {
        m_translation = translation;
        return *this;
    }

    WorldTransformBuilder &with_rotation(const glm::vec3 &rotation) {
        m_rotation = rotation;
        return *this;
    }

    WorldTransformBuilder &with_scale(const glm::vec3 &scale) {
        m_scale = scale;
        return *this;
    }

    WorldTransformBuilder &with_uniform_scale(float scale) {
        m_scale = glm::vec3(scale);
        return *this;
    }

    WorldTransform build() {
        auto object_space_translation = m_object_space_translation.value_or(glm::vec3(0.0, 0.0, 0.0));
        auto translation = m_translation.value_or(glm::vec3(0.0, 0.0, 0.0));
        auto rotation = m_rotation.value_or(glm::vec3(0.0, 0.0, 0.0));
        auto scale = m_scale.value_or(glm::vec3(1.0, 1.0, 1.0));
        auto world = glm::translate(translation) *
                glm::rotate(rotation.z, glm::vec3(0.0, 0.0, 1.0)) *
                glm::rotate(rotation.y, glm::vec3(0.0, 1.0, 0.0)) *
                glm::rotate(rotation.x, glm::vec3(1.0, 0.0, 0.0)) *
                glm::scale(scale) *
                glm::translate(object_space_translation);

        return WorldTransform{world};

    }

private:
    std::optional<glm::vec3> m_object_space_translation;
    std::optional<glm::vec3> m_translation;
    std::optional<glm::vec3> m_rotation;
    std::optional<glm::vec3> m_scale;
};

#endif