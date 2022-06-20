#ifndef RENDERER_CAMERA_CUH_
#define RENDERER_CAMERA_CUH_

#include "ray.cuh"

class RandomGenerator;
/*auto camera = create_device_type<Camera>();*/
class Camera {
public:
    static Camera* create();
    __device__ __host__ void set_position(const glm::vec3 &pos) {
        if(m_position != pos) {
            m_position = pos;
            m_needs_update = true;
        }
    }

    __device__ __host__ void set_direction(const glm::vec3 &dir) {
        if(m_direction != dir) {
            m_direction = dir;
            m_needs_update = true;
        }
    }

    __device__ __host__ void set_up(const glm::vec3 &up) { m_up = up; }

    __device__ __host__ void set_field_of_view(float fov) { m_fov = fov; }

    __device__ __host__ void set_blur_radius(float radius) { m_blur_radius = radius; }

    __device__ __host__ void set_focal_length(float length) { m_focal_length = length; }

    __device__ __host__ void set_shutter_speed(float speed) { m_shutter_speed = speed; }

    __device__ __host__ void set_resolution(const glm::vec2 &res) {
        m_resolution = res;
        m_ratio = m_resolution.x / static_cast<float>(m_resolution.y);
    }

    [[nodiscard]] const glm::vec3 &position() const { return m_position; }

    [[nodiscard]] const glm::vec3 &direction() const { return m_direction; }

    [[nodiscard]] const glm::vec3 &up() const { return m_up; }

    [[nodiscard]] float field_of_view() const { return m_fov; }

    [[nodiscard]] float aspect_ratio() const { return m_ratio; }

    [[nodiscard]] float blur_radius() const { return m_blur_radius; }

    [[nodiscard]] __device__ __host__ float focal_length() const { return m_focal_length; }

    [[nodiscard]] float shutter_speed() const { return m_shutter_speed; }

    [[nodiscard]] bool needs_update() const { return m_needs_update; }

    __device__ __host__ bool update() {
        if(!m_needs_update) {
            return false;
        }
        m_needs_update = false;
        const float distance =  10.0;

        double image_plane_height = 2 * distance * tan(m_fov / 2.0);
        double image_plane_width = image_plane_height * m_ratio;


        glm::vec3 n = m_direction * -1.0f;
        n = glm::normalize(n);

        m_u = glm::cross(m_up, n);
        m_u = glm::normalize(m_u);

        m_v = glm::cross(n, m_u);
        m_v = glm::normalize(m_v);

        glm::vec3 image_plane_center = m_position - (n * distance);

        m_d = glm::length(image_plane_center);

        m_image_plane_origin = image_plane_center +
                               (m_u * (float) (image_plane_width / 2.0)) -
                               (m_v * (float) (image_plane_height / 2.0));
        // m_image_plane_origin = image_plane_center + (m_u * (image_plane_width
        // / 2.0)) - (m_v * (image_plane_height / 2.0));

        m_pixel_width = (float) (image_plane_width / (float) m_resolution.x);
        m_pixel_height = (float) (image_plane_height / (float) m_resolution.y);

        return true;
    }

    [[nodiscard]] __device__ Ray cast_ray(size_t x, size_t y) const;

    [[nodiscard]] __device__ Ray cast_perturbed_ray(
            size_t x,
            size_t y,
            RandomGenerator &random) const;

    [[nodiscard]] glm::vec2 project_onto_image_plane(const glm::vec3 &world_coord) const;

private:
    __device__ __host__ Camera()
            : m_needs_update(true),
              m_position({0, 0, 0}),
              m_direction({0, 0, -1}),
              m_up({0, 1, 0}),
              m_resolution({0, 0}),
              m_fov(Camera::default_fov),
              m_ratio(0),
              m_blur_radius(0),
              m_focal_length(Camera::default_focal_length),
              m_shutter_speed(0) {}

    const float default_fov = 1.047f;
    const float default_focal_length = std::numeric_limits<float>::max();
    bool m_needs_update;
    glm::vec3 m_u;
    glm::vec3 m_v;
    glm::vec3 m_image_plane_origin;

    glm::vec3 m_position;
    glm::vec3 m_direction;
    glm::vec3 m_up;
    glm::uvec2 m_resolution;

    float m_fov;
    float m_ratio;
    float m_pixel_width;
    float m_pixel_height;
    float m_blur_radius;
    float m_focal_length;
    float m_shutter_speed;
    float m_d;
};

#endif