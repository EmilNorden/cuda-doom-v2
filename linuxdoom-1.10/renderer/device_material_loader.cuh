
#ifndef RENDERER_DEVICE_MATERIAL_LOADER_CUH
#define RENDERER_DEVICE_MATERIAL_LOADER_CUH

#include "device_texture_loader.cuh"
#include "device_material.cuh"
#include <string_view>

class DeviceMaterialLoader{
public:
    explicit DeviceMaterialLoader(DeviceTextureLoader& texture_loader) : m_texture_loader(texture_loader) {}
    [[nodiscard]] DeviceMaterial load(std::string_view directory);
private:
    DeviceTextureLoader &m_texture_loader;
};

#endif //RENDERER_DEVICE_MATERIAL_LOADER_CUH
