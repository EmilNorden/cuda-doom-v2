#ifndef DOOM_RT_MATERIAL_CUH_
#define DOOM_RT_MATERIAL_CUH_

#include "rt_init_options.cuh"
#include "renderer/device_material.cuh"
#include "wad/graphics_data.cuh"
#include <string_view>

class DeviceTexture;

void RT_InitMaterials(const RayTracingInitOptions &options);

DeviceMaterial RT_GetMaterial(std::string_view name, DeviceTexture* texture); // TODO: RT_GetMaterial should create the texture

#endif