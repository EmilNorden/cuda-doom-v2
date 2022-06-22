#ifndef RT_RAYTRACING_SCENE_CUH_
#define RT_RAYTRACING_SCENE_CUH_

#include "wad/graphics_data.cuh"

class Scene;
namespace device {

}

void RT_BuildScene(wad::Wad &wad, wad::GraphicsData &graphics_data);

#endif