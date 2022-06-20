//
// Created by emil on 2022-06-12.
//

#ifndef DOOM_RT_RAYTRACING_CUH
#define DOOM_RT_RAYTRACING_CUH

#include "doomtype.h"

void RT_Init();
void RT_BuildScene();
void RT_Enable();
void RT_Disable();
bool RT_IsEnabled();
void RT_RenderSample();
void RT_Present();
void RT_UpdatePalette(byte* palette);

#endif //DOOM_RT_RAYTRACING_CUH
