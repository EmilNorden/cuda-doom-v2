#include "map_thing.cuh"
#include "ray.cuh"

__device__ bool intersects_map_thing(const Ray &ray, MapThing *thing, float &hit_distance, float &u, float &v) {
    auto thing_pos = thing->position;
    auto direction_to_viewer = glm::normalize(glm::vec3(ray.origin().x, 0, ray.origin().z)- thing_pos);
    //auto direction_to_viewer = glm::normalize((thing_pos + glm::vec3(0, thing.max_size.y, 0)) - ray.origin());
    auto tangent = glm::vec3(0, -1, 0);
    auto bitangent = glm::normalize(glm::cross(tangent, direction_to_viewer));

    auto top_left = thing_pos - (bitangent * (thing->max_size.x * 0.5f)); // TODO: This shouldnt be max size, it should be the size of the actual sprite. Perhaps pre-cache for each frame/rotation so we dont have to do lookups?
    top_left.y += thing->max_size.y;

    auto uv_scale = glm::vec2(1,1) / thing->max_size;

    Square s(top_left, bitangent * thing->max_size.x, tangent * thing->max_size.y, uv_scale, nullptr);

    glm::vec3 normal;
    return intersects_wall(ray, &s, hit_distance, u, v, normal);
}