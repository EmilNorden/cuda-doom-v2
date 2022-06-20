#ifndef MAP_THING_CUH_
#define MAP_THING_CUH_

#include "aabb.cuh"
#include "square.cuh"
#include "device_sprite.cuh"

struct MapThing {
    glm::vec2 max_size;
    glm::vec3 position;

    int frame;
    int rotation;
    DeviceSprite sprite;

    MapThing(glm::vec2 max, glm::vec3 pos, int frame, int rotation, DeviceSprite sprite)
            : max_size(max), position(pos), frame(frame), rotation(rotation), sprite(sprite) {

    }
};

__device__ bool intersects_map_thing(const Ray &ray, MapThing *thing, float &hit_distance, float &u, float &v);

#endif