#ifndef RENDERER_INTERSECTION_CUH
#define RENDERER_INTERSECTION_CUH

struct Intersection {
    Intersection() = default;
    DeviceTexture *texture;
    float u{};
    float v{};
    float distance{};
};

#endif