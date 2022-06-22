#ifndef DEVICE_SPRITE_CUH_
#define DEVICE_SPRITE_CUH_

#include "device_texture.cuh"
// #include "../wad/sprites.cuh"
#include <array>

class DeviceSpriteFrame {
public:
    explicit DeviceSpriteFrame(std::array<DeviceTexture*, 8> &rotations) {
        std::copy(std::begin(rotations), std::end(rotations), m_rotations);
    }

    __device__ inline DeviceTexture* get_texture(int rotation) {
        assert(rotation < 8);
        return m_rotations[rotation];
    }
private:
    DeviceTexture* m_rotations[8];
};

class DeviceSprite {
public:
    explicit DeviceSprite(const std::vector<DeviceSpriteFrame> &frames);

    __device__ inline  DeviceTexture *get_texture(int frame, int rotation) {
        assert(frame < m_frame_count);
        return m_frames[frame].get_texture(rotation);
    }

    // TODO: Only for debugging purposes
    [[nodiscard]] size_t frame_count() const { return m_frame_count; }

private:
    DeviceSpriteFrame *m_frames{};
    size_t m_frame_count;
};

#endif