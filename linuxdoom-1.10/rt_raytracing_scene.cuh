#ifndef RT_RAYTRACING_SCENE_CUH_
#define RT_RAYTRACING_SCENE_CUH_

#include "wad/graphics_data.cuh"
#include <unordered_map>
#include <vector>
#include "r_defs.h"
#include "renderer/square.cuh"
#include "renderer/triangle.cuh"

class Scene;
namespace device {

}

struct CeilingWall {
    Square *wall;
    float adjacent_ceiling_height; // Does this need to be a pointer to sector_t to get the "fresh" value in case adjacent sectors update simultaneously?
};

struct FloorWall {
    Square *wall;
    float adjacent_floor_height;
};

struct MovableSector {
    std::vector<CeilingWall> ceiling_walls;
    std::vector<Square*> middle_walls;
    std::vector<FloorWall> floor_walls;
    std::vector<Triangle *> ceiling;
    std::vector<Triangle *> floor;
};

struct BuildSceneResult {
    Scene *scene;
    std::unordered_map<sector_t *, MovableSector> movable_sectors;
};

BuildSceneResult RT_BuildScene(wad::Wad &wad, wad::GraphicsData &graphics_data);

#endif