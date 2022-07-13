#include "scene_entity.cuh"
#include "ray.cuh"

__device__ bool intersects_scene_entity(const Ray &ray, SceneEntity *entity, float &hit_distance, float &u, float &v) {
    auto sprite_texture = entity->sprite.get_material(entity->frame, entity->rotation)->diffuse_map();
    auto sprite_offsets = entity->sprite.get_offsets(entity->frame, entity->rotation);
    auto width = static_cast<float>(sprite_texture->width());
    auto height = static_cast<float>(sprite_texture->height());

    auto thing_pos = entity->position;
    auto direction_to_viewer = glm::normalize(glm::vec3(ray.origin().x, 0, ray.origin().z)- thing_pos);
    auto tangent = glm::vec3(0, -1, 0);
    auto bitangent = glm::normalize(glm::cross(tangent, direction_to_viewer));

    auto top_left = thing_pos - (bitangent * (width * 0.5f));
    top_left.y += height;


    auto uv_scale = glm::vec2(1.0f / width, 1.0f / height);

    Square s(top_left, bitangent * width, tangent * height, uv_scale, DeviceMaterial(nullptr));

    glm::vec3 normal;
    auto result = intersects_wall(ray, &s, hit_distance, u , v, normal);
    return result;
}