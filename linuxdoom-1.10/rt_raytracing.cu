#include "rt_raytracing.cuh"
#include <iostream>
#include <GL/glew.h>
#include "renderer/renderer.cuh"
#include "renderer/cuda_utils.cuh"
#include "renderer/camera.cuh"
#include "renderer/device_random.cuh"
#include "renderer/scene.cuh"
#include "rt_raytracing_opengl.cuh"
#include "opengl/common.h"

// CUDA <-> OpenGL interop
namespace device {
    GLuint opengl_tex_cuda;
    Renderer *renderer;
    Camera *camera;
    Scene *scene;
    RandomGeneratorPool *random;
    std::uint8_t *palette;
}


bool ray_tracing_enabled;

void print_cuda_device_info();
void init_gl_buffers();


void RT_Init() {
    print_cuda_device_info();
    init_gl_buffers();

    cudaDeviceSetLimit(cudaLimitStackSize, 2048);

    device::renderer = new Renderer(device::opengl_tex_cuda, 320, 240);
    device::camera = Camera::create();
    std::cout << "Creating random states..." << std::flush;
    device::random = create_device_type<RandomGeneratorPool>(2048 * 256, 682856);
    std::cout << "Done." << std::endl;
    cuda_assert(cudaMalloc(&device::palette, 768));

//std::vector<Square> &walls, std::vector<Triangle> &floors_ceilings, std::vector<MapThing> &map_things
std::vector<Square> walls;
std::vector<Triangle> fc;
std::vector<MapThing> mt;
    device::scene = create_device_type<Scene>(walls, fc, mt);

    RT_InitGl();
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
    device::renderer->render(
            device::camera,
            device::scene,
            device::random,
            device::palette, nullptr /*This should be needed anymore, just fix sky texture */, 320, 240, 0);

}

void RT_Present() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, device::opengl_tex_cuda);

    RT_RenderQuad();
}

void RT_UpdatePalette(byte* palette) {
    cuda_assert(cudaMemcpy(device::palette, palette, 768, cudaMemcpyHostToDevice));
}