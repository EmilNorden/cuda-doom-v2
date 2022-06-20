#include "device_sprite.cuh"
#include "cuda_utils.cuh"

DeviceSprite::DeviceSprite(const std::vector<DeviceSpriteFrame> &frames) {
    transfer_vector_to_device_memory(frames, &m_frames);
    m_frame_count = frames.size();
}
