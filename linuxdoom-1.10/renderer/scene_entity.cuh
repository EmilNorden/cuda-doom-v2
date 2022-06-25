#ifndef MAP_THING_CUH_
#define MAP_THING_CUH_

#include "aabb.cuh"
#include "square.cuh"
#include "device_sprite.cuh"

struct SceneEntity {
    glm::vec3 position;

    int frame;
    int rotation;
    DeviceSprite sprite;

    SceneEntity(glm::vec3 pos, int frame, int rotation, DeviceSprite sprite)
            : position(pos), frame(frame), rotation(rotation), sprite(sprite) {

    }
};

__device__ bool intersects_scene_entity(const Ray &ray, SceneEntity *entity, float &hit_distance, float &u, float &v);

#endif