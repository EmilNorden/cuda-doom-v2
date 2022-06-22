#ifndef DEVICE_SPRITE_CUH_
#define DEVICE_SPRITE_CUH_

#include "device_texture.cuh"
// #include "../wad/sprites.cuh"
#include <array>

class DeviceSpriteFrame {
public:
    explicit DeviceSpriteFrame(std::array<DeviceTexture*, 8> &rotations, std::array<glm::i16vec2, 8> &offsets) {
        std::copy(std::begin(rotations), std::end(rotations), m_rotations);
        std::copy(std::begin(offsets), std::end(offsets), m_offsets);
    }

    __device__ inline DeviceTexture* get_texture(int rotation) {
        assert(rotation < 8);
        return m_rotations[rotation];
    }

    __device__ inline glm::i16vec2 get_offsets(int rotation) {
        assert(rotation < 8);
        return m_offsets[rotation];
    }

private:
    DeviceTexture* m_rotations[8];
    glm::i16vec2 m_offsets[8];
};

class DeviceSprite {
public:
    explicit DeviceSprite(const std::vector<DeviceSpriteFrame> &frames);

    __device__ inline  DeviceTexture *get_texture(int frame, int rotation) {
        assert(frame < m_frame_count);
        return m_frames[frame].get_texture(rotation);
    }

    __device__ inline glm::i16vec2 get_offsets(int frame, int rotation) {
        assert(frame < m_frame_count);
        return m_frames[frame].get_offsets(rotation);
    }

    // TODO: Only for debugging purposes
    [[nodiscard]] size_t frame_count() const { return m_frame_count; }

private:
    DeviceSpriteFrame *m_frames{};
    size_t m_frame_count;
};

#endif