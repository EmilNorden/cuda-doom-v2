//
// Created by emil on 2021-05-09.
//

#include "renderer.cuh"
#include "camera.cuh"
#include "scene.cuh"

#include "cuda_utils.cuh"
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "device_random.cuh"
#include "transform.cuh"
#include "geometry_helpers.cuh"
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/norm.hpp> // for length2

struct EmissiveSurface {
    // incoming emission
    glm::vec3 incoming_direction{};
    glm::vec3 incoming_emission{};

    // inherent emission
    glm::vec3 emission{};

    // surface data
    glm::vec3 world_space_normal{};
    glm::vec3 world_coordinate{};
    glm::vec3 diffuse_color;
    float roughness{};
    // SceneEntity *entity; // Remove this later and compare intersections using coordinates or something
};

template<size_t Length>
struct LightPath {
    EmissiveSurface surfaces[Length];
    size_t surface_count{};
};

__device__ float matte_brdf(const glm::vec3 &incoming, const ::glm::vec3 &outgoing, const glm::vec3 &surface_normal) {
    auto half = (incoming + outgoing) / 2.0f;
    auto theta = glm::dot(surface_normal, half);
    if (theta < 0) {
        return 0;
    }

    return theta;
}

__device__ glm::vec3
generate_unit_vector_in_cone(const glm::vec3 &cone_direction, float cone_angle, RandomGenerator &random) {

    // Find a tangent orthogonal to cone direction
    auto tangent = glm::vec3{1, 0, 0};
    if (glm::dot(tangent, cone_direction) > 0.99f) {
        tangent = glm::vec3{0, 0, 1};
    }
    tangent = glm::cross(cone_direction, tangent);

    // Now that we have an orthogonal tangent. Rotate it a random direction around
    // The cone direction.
    tangent = glm::rotate(tangent, random.value() * glm::two_pi<float>(), cone_direction);

    // Now, rotate cone_direction around the tangent :)
    return glm::rotate(cone_direction, (1.0f - (random.value() * 2.0f)) * cone_angle, tangent);
}

template<int N>
__device__ glm::vec3
trace_ray(const Ray &ray, Scene *scene, RandomGenerator &random, int depth, std::uint8_t *palette) {
    if (depth == 0) {
        return glm::vec3(1, 1, 0);
    }

    Intersection intersection;
    std::uint8_t palette_index;
    if (scene->intersect(ray, intersection)) {

        palette_index = intersection.texture->sample({intersection.u, intersection.v});
    } else {
        auto pitch = glm::half_pi<float>() - glm::asin(-ray.direction().y);
        auto yaw = fabs(std::atan2(ray.direction().x, ray.direction().z));
        palette_index = scene->sky()->sample({yaw, pitch / glm::pi<float>()});
    }

    return {
            palette[palette_index * 3] / 255.0f,
            palette[(palette_index * 3) + 1] / 255.0f,
            palette[(palette_index * 3) + 2] / 255.0f,
    };

    /*Square s(glm::vec3(0, 0, 0), glm::vec3(1000, 0, 0), glm::vec3(0, -100, 0), 0);

    float f =0.0f;
    float u = 0.0f;
    float v = 0.0f;
    if(intersects_wall(ray, s,f, u, v)) {
        return {1,1,1};
    }*/



    // SceneEntity *entity = nullptr;
    /*
    if (scene->intersect(ray, intersection, &entity)) {


        auto intersection_coordinate = ray.origin() + (ray.direction() * intersection.distance);
        auto &material = entity->mesh()->material();
        float w = 1.0f - intersection.u - intersection.v;

        auto texcoord0 = entity->mesh()->texture_coordinates()[intersection.i0];
        auto texcoord1 = entity->mesh()->texture_coordinates()[intersection.i1];
        auto texcoord2 = entity->mesh()->texture_coordinates()[intersection.i2];

        auto texture_uv = texcoord0 * w + texcoord1 * intersection.u + texcoord2 * intersection.v;

        auto palette_index = material.sample_diffuse(texture_uv);
        return {
                palette[(palette_index*3)] / 255.0f,
                palette[(palette_index*3) + 1] / 255.0f,
                palette[(palette_index*3) + 2] / 255.0f};
    } else {
        return {1,0,1};
    }
*/

    return {1, 0, 1};

}

__global__ void
cudaRender(float *g_odata, Camera *camera, Scene *scene, RandomGeneratorPool *random_pool, std::uint8_t *palette, int width, int height,
           size_t sample) {
    constexpr int PathLength = 1;

    auto tx = threadIdx.x;
    auto ty = threadIdx.y;
    auto bw = blockDim.x;
    auto bh = blockDim.y;
    auto x = blockIdx.x * bw + tx;
    auto y = blockIdx.y * bh + ty;

    auto threads_per_block = bw * bh;
    auto thread_num_in_block = tx + bw * ty;
    auto block_num_in_grid = blockIdx.x + gridDim.x * blockIdx.y;
    auto global_thread_id = block_num_in_grid * threads_per_block + thread_num_in_block;
    // auto global_block_id = block_num_in_grid;
    auto random = random_pool->get_generator(global_thread_id);

    if (x < width && y < height) {
        auto ray = camera->cast_perturbed_ray(x, y, random);

        auto color = trace_ray<PathLength>(ray, scene, random, 3, palette);
        color = glm::clamp(color, {0, 0, 0}, {1, 1, 1});

        glm::vec3 previous_color;
        auto pixel_index = y * (width * 4) + (x * 4);
        g_odata[pixel_index] = ((g_odata[pixel_index] * (float) sample) + color.x) / (sample + 1.0f);
        g_odata[pixel_index + 1] = ((g_odata[pixel_index + 1] * (float) sample) + color.y) / (sample + 1.0f);
        g_odata[pixel_index + 2] = ((g_odata[pixel_index + 2] * (float) sample) + color.z) / (sample + 1.0f);
        g_odata[pixel_index + 3] = 1.0;
    }

}

void Renderer::render(Camera *camera, Scene *scene, RandomGeneratorPool *random, std::uint8_t *palette, int width, int height, size_t sample) {
    dim3 block(16, 16, 1);
    dim3 grid(std::ceil(width / (float) block.x), std::ceil(height / (float) block.y), 1);
    cudaRender<<<grid, block>>>((float *) m_cuda_render_buffer, camera, scene, random, palette, width,
                                height, sample);

    cudaArray *texture_ptr;
    cuda_assert(cudaGraphicsMapResources(1, &m_cuda_tex_resource, 0));
    cuda_assert(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_cuda_tex_resource, 0, 0));

    // TODO: Havent we already calculated this?
    auto size_tex_data = sizeof(GLfloat) * width * height * 4;
    cuda_assert(cudaMemcpyToArray(texture_ptr, 0, 0, m_cuda_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
    cuda_assert(cudaGraphicsUnmapResources(1, &m_cuda_tex_resource, 0));
}

Renderer::Renderer(GLuint gl_texture, int width, int height)
        : m_cuda_render_buffer(nullptr) {
    allocate_render_buffer(width, height);

    cuda_assert(cudaGraphicsGLRegisterImage(&m_cuda_tex_resource, gl_texture, GL_TEXTURE_2D,
                                            cudaGraphicsRegisterFlagsWriteDiscard));
}

void Renderer::allocate_render_buffer(int width, int height) {
    if (m_cuda_render_buffer) {
        cudaFree(m_cuda_render_buffer);
    }

    auto buffer_size = width * height * 4 * sizeof(GLfloat); // Is GLubyte ever larger than 1?
    cuda_assert(cudaMalloc(&m_cuda_render_buffer, buffer_size));
}

Renderer::~Renderer() {
    if (m_cuda_render_buffer) {
        cudaFree(m_cuda_render_buffer);
    }
}

void Renderer::render(int width, int height, const Camera &camera, const Scene &scene) {

}