#ifndef DOOM_RT_INIT_OPTIONS_CUH_
#define DOOM_RT_INIT_OPTIONS_CUH_

#include <optional>
#include <filesystem>

struct RayTracingInitOptions  {
    char **wadfiles;
    std::optional<std::filesystem::path> materials_file;
};

#endif