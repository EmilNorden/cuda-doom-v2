#ifndef RENDERER_BRDF_CUH_
#define RENDERER_BRDF_CUH_

#include <glm/glm.hpp>

enum class BrdfType {
    Lambertian
    CookTorrance
};

float brdf(BrdfType type, const glm::vec3& incoming, const glm::vec3& outgoing, const glm::vec3& surface_normal) {
    switch(type) {
        case BrdfType::Lambertian:
            // Lambertian is isotropic, uniform reflectance in all directions regardless of the observers angle.
            // I guess this is why its only calculated using incoming and surface normal. Outgoing direction does not matter.
            return glm::abs(glm::dot(glm::vec3(-incoming.x, -incoming.y, -incoming.z), surface_normal));
            break;
        case BrdfType::CookTorrance:
            return 0.0f;
            break;
    }
}

#endif