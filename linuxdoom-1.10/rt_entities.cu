#include "rt_entities.cuh"
#include "rt_raytracing.cuh"
#include "renderer/device_sprite.cuh"
#include "renderer/device_texture.cuh"
#include "renderer/cuda_utils.cuh"
#include "renderer/scene_entity.cuh"
#include "r_main.h"
#include "rt_material.cuh"
#include "renderer/scene.cuh"

namespace detail {
    std::array<std::optional<DeviceSprite>, NUMSPRITES> device_sprite_cache;
    std::vector<SceneEntity *> scene_entities_to_free;
    bool bulk_attach_mode = false;
    std::vector<SceneEntity *> bulk_attach_entities;
}


std::optional<DeviceSprite> RT_GetDeviceSprite(spritenum_t sprite);

SceneEntity *RT_CreateMapThing(mobjtype_t type, mobj_t *obj) {
    auto info = &mobjinfo[type];
    if (info->flags & MF_NOSECTOR) {
        // Is invisible
        return nullptr;
    }

    auto spawn_state = &states[info->spawnstate];

    auto sprite = RT_GetDeviceSprite(spawn_state->sprite);


    if (!sprite.has_value()) {
        std::cerr << "Couldnt find sprite " << spawn_state->sprite << "!\n";
        return nullptr;
    }

    auto frame = spawn_state->frame;
    auto position = glm::vec3(
            RT_FixedToFloating(obj->x),
            RT_FixedToFloating(obj->z),
            RT_FixedToFloating(obj->y));
    return create_device_type<SceneEntity>(position,
                                           frame & FF_FRAMEMASK,
                                           0,
                                           sprite.value(),
                                           type == MT_PLAYER);
}

void RT_DestroySceneEntity(SceneEntity *entity) {
    if (!entity) {
        return;
    }

    detail::scene_entities_to_free.push_back(entity);
    // cudaFree(entity);
}

void RT_UpdateEntityPosition(mobj_t *obj) {
    if (!obj->scene_entity) {
        return;
    }

    obj->scene_entity->sprite = RT_GetDeviceSprite(obj->sprite).value();
    obj->scene_entity->frame = obj->frame & FF_FRAMEMASK;
    auto ang = R_PointToAngle(obj->x, obj->y);
    auto rot = (ang - obj->angle + (unsigned) (ANG45 / 2) * 9) >> 29;
    obj->scene_entity->rotation = rot;

    auto new_position = glm::vec3{
            RT_FixedToFloating(obj->x),
            RT_FixedToFloating(obj->z),
            RT_FixedToFloating(obj->y)
    };

    if(new_position == obj->scene_entity->position) {
        return;
    }
    obj->scene_entity->position = new_position;

    device::scene->refresh_entity(obj->scene_entity);
}

std::optional<DeviceSprite> RT_GetDeviceSprite(spritenum_t s) {
    if (detail::device_sprite_cache[s].has_value()) {
        return detail::device_sprite_cache[s];
    }

    auto sprite = detail::sprite_data->sprites()[s];
    std::vector<DeviceSpriteFrame> device_sprite_frames;
    for (auto &sprite_frame: sprite.frames) {
        std::array<DeviceMaterial, 8> rotation_materials{};
        std::array<glm::i16vec2, 8> texture_offsets{};
        if (sprite_frame.rotate) {
            for (int rot = 0; rot < 8; ++rot) {
                auto sprite_lump = detail::sprite_data->sprite_lumps_start() +
                                   sprite_frame.lumps[rot]; // TODO: Store lump name in frame.lumps[] instead?
                auto sprite_lump_name = detail::wad->get_lump_name(sprite_lump);

                const auto &picture = detail::graphics_data->get_sprite(sprite_lump_name);

                texture_offsets[rot] = glm::i16vec2(picture.left_offset, picture.top_offset);

                if (sprite_frame.flip[rot]) {
                    auto flipped = wad::flip_picture(picture);
                    rotation_materials[rot] = RT_GetMaterial(sprite_lump_name,
                                                             create_device_type<DeviceTexture>(flipped.pixels,
                                                                                               flipped.width,
                                                                                               flipped.height));
                } else {
                    rotation_materials[rot] = RT_GetMaterial(sprite_lump_name,
                                                             create_device_type<DeviceTexture>(picture.pixels,
                                                                                               picture.width,
                                                                                               picture.height));
                }
            }
        } else {
            auto sprite_lump = detail::sprite_data->sprite_lumps_start() +
                               sprite_frame.lumps[0]; // TODO: Store lump name in frame.lumps[] instead?
            auto sprite_lump_name = detail::wad->get_lump_name(sprite_lump);

            const auto &picture = detail::graphics_data->get_sprite(sprite_lump_name);

            for (int rot = 0; rot < 8; ++rot) {
                rotation_materials[rot] = RT_GetMaterial(sprite_lump_name,
                                                         create_device_type<DeviceTexture>(picture.pixels,
                                                                                           picture.width,
                                                                                           picture.height));
                texture_offsets[rot] = glm::i16vec2(picture.left_offset, picture.top_offset);
            }
        }


        device_sprite_frames.emplace_back(rotation_materials, texture_offsets);
    }

    DeviceSprite device_sprite(device_sprite_frames);

    detail::device_sprite_cache[s] = device_sprite;

    return device_sprite;
}

void RT_AttachToScene(SceneEntity *entity) {
    if (!entity) {
        return;
    }

    if (detail::bulk_attach_mode) {
        detail::bulk_attach_entities.push_back(entity);
    } else {
        device::scene->add_entity(entity);
    }
}

bool RT_DetachFromScene(SceneEntity *entity) {
    if (!entity) {
        return false;
    }

    return device::scene->remove_entity(entity);
}

void RT_BeginAttach() {
    if (detail::bulk_attach_mode) {
        std::cerr << "RT_BeginAttach called without calling RT_EndAttach\n";
        return;
    }

    detail::bulk_attach_mode = true;
}

void RT_EndAttach() {
    if (!detail::bulk_attach_mode) {
        std::cerr << "RT_EndAttach called without calling RT_BeginAttach\n";
        return;
    }

    detail::bulk_attach_mode = false;

    printf("Bulk attaching %zu entities\n", detail::bulk_attach_entities.size());

    device::scene->add_entities(detail::bulk_attach_entities);
    detail::bulk_attach_entities.clear();
}