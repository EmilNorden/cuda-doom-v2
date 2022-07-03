#include "rt_texture.cuh"

#include "renderer/device_texture.cuh"
#include <unordered_map>

class TextureCache {
public:
    DeviceTexture *get_texture_for_flat(short flatnum) {
        auto it = m_flat_textures.find(flatnum);
        if (it != m_flat_textures.end()) {
            return it->second;
        }
        return nullptr;
    }

    DeviceTexture *get_texture(short texture_number) {
        auto it = m_textures.find(texture_number);
        if (it != m_textures.end()) {
            return it->second;
        }
        return nullptr;
    }

    void insert_flat_texture(short flatnum, DeviceTexture *texture) {
        m_flat_textures[flatnum] = texture;
    }

    void insert_texture(short texture_number, DeviceTexture *texture) {
        m_textures[texture_number] = texture;
    }

private:
    std::unordered_map<short, DeviceTexture *> m_flat_textures;
    std::unordered_map<short, DeviceTexture *> m_textures;

};