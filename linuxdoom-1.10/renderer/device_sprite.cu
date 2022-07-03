#include "device_sprite.cuh"
#include "cuda_utils.cuh"

DeviceSprite::DeviceSprite(const std::vector<DeviceSpriteFrame> &frames) {
    m_has_emissive_frames = false;
    for(auto &frame : frames) {
        if(frame.has_emissive_material()) {
            m_has_emissive_frames = true;
        }
    }
    transfer_vector_to_device_memory(frames, &m_frames);
    m_frame_count = frames.size();
}
