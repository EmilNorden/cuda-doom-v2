#ifndef DEVICE_SPRITE_CUH_
#define DEVICE_SPRITE_CUH_

#include "device_texture.cuh"
#include "device_material.cuh"
#include <array>
#include <algorithm>

class DeviceSpriteFrame {
public:
    explicit DeviceSpriteFrame(std::array<DeviceMaterial, 8> &rotations, std::array<glm::i16vec2, 8> &offsets) {
        std::copy(std::begin(rotations), std::end(rotations), m_rotations);
        std::copy(std::begin(offsets), std::end(offsets), m_offsets);
    }

    __device__ inline DeviceMaterial *get_material(int rotation) {
        assert(rotation < 8);
        return &m_rotations[rotation];
    }

    __device__ inline glm::i16vec2 get_offsets(int rotation) {
        assert(rotation < 8);
        return m_offsets[rotation];
    }

    __host__ glm::vec2 calculate_max_bounds() const {
        glm::vec2 max = glm::vec2(m_rotations[0].diffuse_map()->width(), m_rotations[0].diffuse_map()->height());

        for (int i = 1; i < 8; ++i) {
            max = glm::max(max,
                           glm::vec2(m_rotations[i].diffuse_map()->width(), m_rotations[i].diffuse_map()->height()));
        }

        return max;
    }

    [[nodiscard]] glm::vec3 calculate_max_emission() const {
        auto max = m_rotations[0].emission();
        for(int i = 1; i < 8; ++i) {
            max = glm::max(max, m_rotations[i].emission());
        }

        return max;
    }

    bool has_emissive_material() const {
        for (auto & mat : m_rotations) {
            if (mat.emission().x > 0 || mat.emission().y > 0 || mat.emission().z > 0) {
                return true;
            }

            return false;
        }
        /*return std::any_of(std::begin(m_rotations), std::end(m_rotations), [](DeviceMaterial &mat) {
            return mat.emission().x > 0 || mat.emission().y > 0 || mat.emission().z > 0;
        });*/
    }

private:
    DeviceMaterial m_rotations[8];
    glm::i16vec2 m_offsets[8];
};

class DeviceSprite {
public:
    explicit DeviceSprite(const std::vector<DeviceSpriteFrame> &frames);

    __device__ inline DeviceMaterial *get_material(int frame, int rotation) {
        assert(frame < m_frame_count);
        return m_frames[frame].get_material(rotation);
    }

    __device__ inline glm::i16vec2 get_offsets(int frame, int rotation) {
        assert(frame < m_frame_count);
        return m_frames[frame].get_offsets(rotation);
    }

    __host__ glm::vec2 calculate_max_bounds() const {
        auto max = m_frames[0].calculate_max_bounds();

        for (int i = 0; i < m_frame_count; ++i) {
            max = glm::max(max, m_frames[i].calculate_max_bounds());
        }

        return max;
    }

    [[nodiscard]] glm::vec3 calculate_max_emission() const {
        auto max = m_frames[0].calculate_max_emission();
        for (int i = 0; i < m_frame_count; ++i) {
            max = glm::max(max, m_frames[i].calculate_max_emission());
        }

        return max;
    }

    // TODO: Only for debugging purposes
    [[nodiscard]] size_t frame_count() const { return m_frame_count; }

    [[nodiscard]] bool has_emissive_frames() const {
        return m_has_emissive_frames;
    }

private:
    DeviceSpriteFrame *m_frames{};
    size_t m_frame_count;
    bool m_has_emissive_frames;
};

#endif