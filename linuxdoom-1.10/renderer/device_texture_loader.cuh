#ifndef RENDERER_DEVICE_TEXTURE_LOADER_CUH_
#define RENDERER_DEVICE_TEXTURE_LOADER_CUH_

#include <string>

class DeviceTexture;

class DeviceTextureLoader {
public:
    DeviceTexture *load(const std::string &path);
    static bool file_is_supported(const std::string &path);
private:
};

#endif