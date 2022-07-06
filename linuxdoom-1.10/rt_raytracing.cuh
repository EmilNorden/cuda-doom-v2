//
// Created by emil on 2022-06-12.
//

#ifndef DOOM_RT_RAYTRACING_CUH
#define DOOM_RT_RAYTRACING_CUH

#include "doomtype.h"
#include "d_player.h"
#include "wad/sprites.cuh"
#include "r_defs.h"
#include "rt_init_options.cuh"

class Scene;
class Camera;
namespace device {
    extern Camera *camera;
    extern Scene *scene;
    extern std::uint8_t *palette;
}

namespace detail {
    extern wad::GraphicsData *graphics_data;
    extern wad::SpriteData *sprite_data;
    extern wad::Wad *wad;
}

void RT_Init(RayTracingInitOptions options);

void RT_BuildScene();

void RT_Enable();

void RT_Disable();

bool RT_IsEnabled();

void RT_RenderSample();

void RT_Present();

void RT_UpdatePalette(byte *palette);

void RT_UpdateCameraFromPlayer(player_t *player);

void RT_WindowChanged();

void RT_VerticalDoorChanged(sector_t *sector);

void RT_CeilingChanged(sector_t *sector);

void RT_SectorFloorHeightChanged(sector_t *sector);

int *RT_GetFrameTime();

inline float RT_FixedToFloating(int value) {
    return static_cast<float>(value) / 65536.0f;
}

#endif //DOOM_RT_RAYTRACING_CUH
