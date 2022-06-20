#ifndef RENDERER_SURFACE_DESCRIPTION_CUH_
#define RENDERER_SURFACE_DESCRIPTION_CUH_

class IndexedDeviceMesh;

struct SurfaceDescription {
    IndexedDeviceMesh *mesh;
    glm::vec3 world_coordinate;
    glm::vec3 world_normal;
    glm::vec3 emission;
};

#endif
