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
    bool is_player;

    SceneEntity(glm::vec3 pos, int frame, int rotation, DeviceSprite sprite, bool is_player = false)
            : position(pos), frame(frame), rotation(rotation), sprite(sprite), is_player(is_player) {

    }

    [[nodiscard]] DeviceMaterial *get_current_material() {
        return sprite.get_material(frame, rotation);
    }
};

__device__ bool intersects_scene_entity(const Ray &ray, SceneEntity *entity, float &hit_distance, float &u, float &v);

#endif