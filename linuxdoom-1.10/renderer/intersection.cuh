#ifndef RENDERER_INTERSECTION_CUH
#define RENDERER_INTERSECTION_CUH

struct Intersection {
    Intersection() = default;
    DeviceMaterial *material;
    float u{};
    float v{};
    float distance{};
    glm::vec3 world_normal;
};

#endif