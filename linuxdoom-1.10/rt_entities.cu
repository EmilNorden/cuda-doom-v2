#include "rt_entities.cuh"
#include "rt_raytracing.cuh"
#include "renderer/device_sprite.cuh"
#include "renderer/device_texture.cuh"
#include "renderer/cuda_utils.cuh"
#include "renderer/scene_entity.cuh"
#include "r_main.h"

namespace detail {
    std::array<std::optional<DeviceSprite>, NUMSPRITES> device_sprite_cache;
}

std::optional<DeviceSprite> RT_GetDeviceSprite(spritenum_t sprite);

SceneEntity *RT_CreateMapThing(mobjtype_t type, mobj_t *obj) {
    auto info = &mobjinfo[type];
    if(info->flags & MF_NOSECTOR) {
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
                                           sprite.value());
}

void RT_DestroySceneEntity(SceneEntity *entity) {
    if (!entity) {
        return;
    }

    cudaFree(entity);
}

void RT_UpdateEntityPosition(mobj_t *obj) {
    if (!obj->scene_entity) {
        return;
    }

    obj->scene_entity->position = glm::vec3{
            RT_FixedToFloating(obj->x),
            RT_FixedToFloating(obj->z),
            RT_FixedToFloating(obj->y)
    };

    obj->scene_entity->sprite = RT_GetDeviceSprite(obj->sprite).value();
    obj->scene_entity->frame = obj->frame & FF_FRAMEMASK;
    auto ang = R_PointToAngle(obj->x, obj->y);
    auto rot = (ang - obj->angle + (unsigned) (ANG45 / 2) * 9) >> 29;
    obj->scene_entity->rotation = rot;
}

std::optional<DeviceSprite> RT_GetDeviceSprite(spritenum_t s) {
    if(detail::device_sprite_cache[s].has_value()) {
        return detail::device_sprite_cache[s];
    }

    auto sprite = detail::sprite_data->sprites()[s];
    std::vector<DeviceSpriteFrame> device_sprite_frames;
    for (auto &sprite_frame: sprite.frames) {
        std::array<DeviceTexture *, 8> rotation_textures{};
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
                    rotation_textures[rot] = create_device_type<DeviceTexture>(flipped.pixels, flipped.width,
                                                                               flipped.height);
                } else {
                    rotation_textures[rot] = create_device_type<DeviceTexture>(picture.pixels, picture.width,
                                                                               picture.height);
                }
            }
        } else {
            auto sprite_lump = detail::sprite_data->sprite_lumps_start() +
                               sprite_frame.lumps[0]; // TODO: Store lump name in frame.lumps[] instead?
            auto sprite_lump_name = detail::wad->get_lump_name(sprite_lump);

            const auto &picture = detail::graphics_data->get_sprite(sprite_lump_name);

            for (int rot = 0; rot < 8; ++rot) {
                rotation_textures[rot] = create_device_type<DeviceTexture>(picture.pixels, picture.width,
                                                                           picture.height);
                texture_offsets[rot] = glm::i16vec2(picture.left_offset, picture.top_offset);
            }
        }


        device_sprite_frames.emplace_back(rotation_textures, texture_offsets);
    }

    DeviceSprite device_sprite(device_sprite_frames);

    detail::device_sprite_cache[s] = device_sprite;

    return device_sprite;
}