//
// Created by emil on 2022-06-12.
//

#ifndef DOOM_RT_RAYTRACING_CUH
#define DOOM_RT_RAYTRACING_CUH

#include "doomtype.h"
#include "d_player.h"

void RT_Init(char **wadfiles);
void RT_BuildScene();
void RT_Enable();
void RT_Disable();
bool RT_IsEnabled();
void RT_RenderSample();
void RT_Present();
void RT_UpdatePalette(byte* palette);
void RT_UpdateCameraFromPlayer(player_t *player);
void RT_WindowChanged();

#endif //DOOM_RT_RAYTRACING_CUH
