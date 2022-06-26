#ifndef RT_RAYTRACING_SCENE_CUH_
#define RT_RAYTRACING_SCENE_CUH_

#include "wad/graphics_data.cuh"
#include <unordered_map>
#include <vector>
#include "r_defs.h"

class Scene;
class Square;
class Triangle;
namespace device {

}

struct TopWall {
    Square *wall;
    float adjacent_ceiling_height; // Does this need to be a pointer to sector_t to get the "fresh" value in case adjacent sectors update simultaneously?
};

struct BottomWall {
    Square *wall;
    float adjacent_floor_height;
};

struct SectorGeometry {
    std::vector<TopWall> top_walls;
    std::vector<TopWall> adjacent_top_walls;
    std::vector<Square*> middle_walls;
    std::vector<BottomWall> bottom_walls;
    std::vector<BottomWall> adjacent_bottom_walls;
    std::vector<Triangle *> ceiling;
    std::vector<Triangle *> floor;
};

struct VerticalDoor {
    SectorGeometry geometry;
};

struct BuildSceneResult {
    Scene *scene;
    std::unordered_map<sector_t *, SectorGeometry> sector_geometry;
};

BuildSceneResult RT_BuildScene(wad::Wad &wad, wad::GraphicsData &graphics_data);

void RT_ChangeSideTopTexture(side_t *side, int texture_num);

void RT_ChangeSideMidTexture(side_t *side, int texture_num);

void RT_ChangeSideBottomTexture(side_t *side, int texture_num);

#endif