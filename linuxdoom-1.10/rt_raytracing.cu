#include "rt_raytracing.cuh"
#include <iostream>
#include <GL/glew.h>
#include "renderer/renderer.cuh"
#include "renderer/cuda_utils.cuh"
#include "renderer/camera.cuh"
#include "renderer/device_random.cuh"
#include "renderer/scene.cuh"
#include "rt_raytracing_opengl.cuh"
#include "rt_raytracing_scene.cuh"
#include "opengl/common.h"
#include "wad/graphics_data.cuh"
#include "wad/wad.cuh"
#include "wad/sprites.cuh"
#include "p_spec.h"
#include <glm/gtx/rotate_vector.hpp>

// CUDA <-> OpenGL interop
namespace device {
    GLuint opengl_tex_cuda;
    Renderer *renderer;
    Camera *camera;
    Scene *scene;
    RandomGeneratorPool *random;
    std::uint8_t *palette;
}

namespace detail {
    wad::GraphicsData *graphics_data;
    wad::SpriteData *sprite_data;
    wad::Wad *wad;
    size_t current_sample;
    bool has_pending_entities = false;
    std::vector<SceneEntity *> pending_attach_entities;
    std::vector<SceneEntity *> pending_detach_entities;
    std::vector<SceneEntity *> entities;
    std::unordered_map<sector_t*, MovableSector> movable_sectors;
}


bool ray_tracing_enabled;

void print_cuda_device_info();

void init_gl_buffers();

void RT_ApplyPendingEntities();

void RT_SwapRemoveEntity(std::vector<SceneEntity *> &collection, SceneEntity *entity_to_remove);

void RT_InitGraphics(char **wadfiles) {
    std::vector<std::filesystem::path> paths;
    for (; *wadfiles; wadfiles++) {
        paths.emplace_back(*wadfiles);
    }

    detail::wad = new wad::Wad(paths);
    detail::graphics_data = new wad::GraphicsData(*detail::wad);
    detail::sprite_data = new wad::SpriteData(*detail::wad, sprnames, NUMSPRITES);
}


void RT_Init(char **wadfiles) {

    print_cuda_device_info();
    init_gl_buffers();

    cudaDeviceSetLimit(cudaLimitStackSize, 2048);

    detail::current_sample = 0;
    device::renderer = new Renderer(device::opengl_tex_cuda, 320, 240);
    device::camera = Camera::create();


    //auto camera_position = glm::vec3(-800.0, 20.0, -100.0);
    auto camera_position = glm::vec3(-765.315, 41.1001, -96.0371);// glm::vec3( -645.167, 58.7087, -412.004);
    auto camera_direction = glm::normalize(glm::vec3(0.0, 0.0, 0.0f) - camera_position);
    device::camera->set_position(camera_position);
    device::camera->set_direction(camera_direction);
    device::camera->set_up(glm::vec3(0.0, 1.0, 0.0));
    device::camera->set_field_of_view(75.0 * (3.1415 / 180.0));
    device::camera->set_blur_radius(0.0); // (0.03);
    device::camera->set_focal_length(60.0);
    device::camera->set_shutter_speed(0.0);
    device::camera->set_resolution(glm::vec2(320, 240));
    device::camera->update();
    std::cout << "Creating random states..." << std::flush;
    device::random = create_device_type<RandomGeneratorPool>(2048 * 256, 682856);
    std::cout << "Done." << std::endl;
    cuda_assert(cudaMalloc(&device::palette, 768));

//std::vector<Square> &walls, std::vector<Triangle> &floors_ceilings, std::vector<MapThing> &map_things
    std::vector<Square *> walls;
    std::vector<Triangle *> fc;
    std::vector<SceneEntity *> mt;
    device::scene = create_device_type<Scene>(walls, fc, mt, nullptr);

    RT_InitGl();
    RT_InitGraphics(wadfiles);
}


void print_cuda_device_info() {
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);

    std::cout << "Using the following CUDA device: " << std::endl;

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << (int) error_id << "\n" << cudaGetErrorString(error_id)
                  << std::endl;
        exit(1);
    }

    if (device_count == 0) {
        std::cout << "There are no available devices that support CUDA" << std::endl;
        exit(1);
    }

    int device_id = 0;

    cudaSetDevice(device_id);
    cudaDeviceProp device_properties{};
    cudaGetDeviceProperties(&device_properties, device_id);

    std::cout << "  Name: " << device_properties.name << "\n";

    int driver_version, runtime_version;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    printf("  CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driver_version / 1000,
           (driver_version % 100) / 10, runtime_version / 1000, (runtime_version % 100) / 10);
    printf("  CUDA Capability Major/Minor version number: %d.%d\n", device_properties.major,
           device_properties.minor);
    printf("  SM Count: %d, Warp size: %d, Shared mem/block %zu \n\n", device_properties.multiProcessorCount,
           device_properties.warpSize, device_properties.sharedMemPerBlock);

}

void init_gl_buffers() {
    const int WIDTH = 320;
    const int HEIGHT = 240;
    glGenTextures(1, &device::opengl_tex_cuda);
    glBindTexture(GL_TEXTURE_2D, device::opengl_tex_cuda);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);

    check_for_gl_errors();
}

void RT_Enable() {
    ray_tracing_enabled = true;
}

void RT_Disable() {
    ray_tracing_enabled = false;
}

bool RT_IsEnabled() {
    return ray_tracing_enabled;
}

void RT_RenderSample() {
    if (detail::has_pending_entities) {
        RT_ApplyPendingEntities();
    }

    device::renderer->render(
            device::camera,
            device::scene,
            device::random,
            device::palette,
            320,
            240,
            0);
    detail::current_sample++;
}

void RT_Present() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, device::opengl_tex_cuda);

    RT_RenderQuad();
}

void RT_UpdatePalette(byte *palette) {
    cuda_assert(cudaMemcpy(device::palette, palette, 768, cudaMemcpyHostToDevice));
}

void RT_BuildScene() {
    if (device::scene) {
        cudaFree(device::scene);
        device::scene = nullptr;
    }

    auto result = RT_BuildScene(*detail::wad, *detail::graphics_data);
    device::scene = result.scene;
    detail::movable_sectors = result.movable_sectors;
}

void RT_UpdateCameraFromPlayer(player_t *player) {

    auto factor =
            static_cast<float>(player->mo->angle) / static_cast<float>(std::numeric_limits<unsigned>::max()) - 0.25f;
    auto radians = -factor * glm::two_pi<float>();
    auto direction = glm::normalize(glm::vec3(glm::sin(radians), 0, glm::cos(radians)));

    // Z is up
    device::camera->set_position({
                                         static_cast<float>(player->mo->x) / 65536.0f,
                                         static_cast<float>(player->viewz) / 65536.0f,
                                         static_cast<float>(player->mo->y) / 65536.0f});

    device::camera->set_direction(direction);
    if (device::camera->update()) {
        detail::current_sample = 0;
    }
}

void RT_WindowChanged() {
    init_gl_buffers();
    RT_InitGl();
    device::renderer = new Renderer(device::opengl_tex_cuda, 320, 240);
    printf("GL BUFFERS RECREATED\n");
}

void RT_AttachToScene(SceneEntity *entity) {
    if (!entity) {
        return;
    }

    detail::has_pending_entities = true;
    detail::pending_attach_entities.push_back(entity);
}

void RT_DetachFromScene(SceneEntity *entity) {
    if (!entity) {
        return;
    }

    detail::has_pending_entities = true;
    detail::pending_detach_entities.push_back(entity);
}

void RT_ApplyPendingEntities() {
    detail::has_pending_entities = false;

    for (auto detach_entity: detail::pending_detach_entities) {
        RT_SwapRemoveEntity(detail::entities, detach_entity);
        RT_SwapRemoveEntity(detail::pending_attach_entities, detach_entity);
    }
    detail::pending_detach_entities.clear();

    detail::entities.insert(detail::entities.end(), detail::pending_attach_entities.begin(),
                            detail::pending_attach_entities.end());
    detail::pending_attach_entities.clear();


    device::scene->rebuild_entities(detail::entities);
}

void RT_SwapRemoveEntity(std::vector<SceneEntity*> &collection, SceneEntity *entity_to_remove) {
    auto iter = std::find(collection.begin(), collection.end(), entity_to_remove);
    if(iter == collection.end()) {
        return;
    }

    //std::iter_swap(*iter, collection.back());
    //collection.pop_back();
    collection.erase(iter);
}

void RT_SectorCeilingHeightChanged(sector_t *sector) {
    auto it = detail::movable_sectors.find(sector);
    if(it == detail::movable_sectors.end()) {
        return;
    }

    auto& movable_sector = it->second;
    auto door = (vldoor_t*)sector->specialdata;

    auto ceiling_height = RT_FixedToFloating(sector->ceilingheight);
    auto door_total_height = RT_FixedToFloating(door->topheight) - RT_FixedToFloating(sector->floorheight);

    // Actual door
    for(auto wall : movable_sector.ceiling_walls) {
        wall.wall->vertical_len = wall.adjacent_ceiling_height - ceiling_height;
        wall.wall->uv_offset = ceiling_height - RT_FixedToFloating(sector->floorheight); // door_total_height - (wall.adjacent_ceiling_height - ceiling_height);
    }


    for(auto wall : movable_sector.floor_walls) {
        wall.wall->top_left.y = ceiling_height;
        wall.wall->vertical_len = ceiling_height- wall.adjacent_floor_height;
    }

    // Side walls, ie door frame.
    for(auto wall : movable_sector.middle_walls) {
        wall->top_left.y = RT_FixedToFloating(door->topheight);
        wall->vertical_len = door_total_height;
        wall->vertical_vec = {0.0f, -1.0f, 0.0f};
        wall->uv_scale.y = (wall->vertical_len / wall->texture->height()) / wall->vertical_len;
    }

    for(auto ceiling : movable_sector.ceiling) {
        ceiling->v0.y = ceiling->v1.y = ceiling->v2.y = ceiling_height;
    }
}

void RT_SectorFloorHeightChanged(sector_t *sector) {
    auto it = detail::movable_sectors.find(sector);
    if(it == detail::movable_sectors.end()) {
        return;
    }

    auto& movable_sector = it->second;

    auto floor_height = RT_FixedToFloating(sector->floorheight);
    for(auto wall : movable_sector.floor_walls) {
        wall.wall->top_left.y = floor_height;

        wall.wall->vertical_len = glm::abs(wall.adjacent_floor_height - floor_height);
    }


    for(auto wall: movable_sector.middle_walls) {
        wall->vertical_len = wall->top_left.y - floor_height;
        wall->vertical_vec = {0.0f, -1.0f, 0.0f};
        wall->uv_scale.y = (wall->vertical_len / wall->texture->height()) / wall->vertical_len;
    }

    for(auto floor : movable_sector.floor) {
        floor->v0.y = floor_height;
        floor->v1.y = floor_height;
        floor->v2.y = floor_height;
    }
}