#ifndef RENDERER_AUTOFOCUS_CUH_
#define RENDERER_AUTOFOCUS_CUH_

class Camera;
class Scene;

void device_autofocus(Camera* camera, Scene* scene, size_t width, size_t height);

#endif