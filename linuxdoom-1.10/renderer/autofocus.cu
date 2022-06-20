#include "autofocus.cuh"
#include "camera.cuh"
#include "scene.cuh"

__global__ void autofocus_kernel(Camera* camera, Scene* scene, size_t width, size_t height){

    auto image_center_x = width / 2;
    auto image_center_y = height / 2;

    auto ray = camera->cast_ray(image_center_x, image_center_y);

    float closest_intersection = 1000.0f;
    Intersection intersection;
    if(scene->intersect(ray, intersection)) {
        closest_intersection = intersection.distance;
    }

    camera->set_focal_length(closest_intersection);
}

void device_autofocus(Camera* camera, Scene* scene, size_t width, size_t height) {
    autofocus_kernel<<<1, 1>>>(camera, scene, width, height);
    cudaDeviceSynchronize();
}