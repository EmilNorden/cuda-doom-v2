#include "rt_dizzy_effect.cuh"
#include "rt_raytracing.cuh"
#include "renderer/camera.cuh"
#include "doomstat.h"

enum class DizzyState {
    Idle,
    Full,
    Clearing
};

DizzyState current_state = DizzyState::Idle;
int dizzy_timestamp = 0;
constexpr float DizzySize = 0.3f;

void RT_PlayerDamaged(int damage) {
    printf("I took %d damage\n", damage);
    if(damage < 16) {
        return;
    }
    if (current_state != DizzyState::Idle) {
        return;
    }

    device::camera->set_blur_radius(DizzySize);
    device::camera->set_focal_length(10.0f);
    device::camera->update();

    current_state = DizzyState::Full;
    dizzy_timestamp = gametic;
}

void RT_DizzyTick() {
    constexpr int dizzy_ticks = 25;
    constexpr int clearing_ticks = 50;
    switch (current_state) {
        case DizzyState::Full:
            printf("gametic is now %d\n", gametic);
            if (gametic - dizzy_timestamp >= dizzy_ticks) {
                current_state = DizzyState::Clearing;
                dizzy_timestamp = gametic;
            }
            break;
        case DizzyState::Idle:
            break;
        case DizzyState::Clearing: {
            if (gametic - dizzy_timestamp >= clearing_ticks) {
                device::camera->set_blur_radius(0);
                current_state = DizzyState::Idle;
            } else {
                auto progress = static_cast<float>(gametic - dizzy_timestamp) / static_cast<float>(clearing_ticks);
                device::camera->set_blur_radius((1.0f - progress) * DizzySize);
            }
            device::camera->update();

            break;
        }

    }
}