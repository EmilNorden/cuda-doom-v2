#ifndef RENDERER_SCENE_H_
#define RENDERER_SCENE_H_

#include <glm/glm.hpp>
#include <optional>
#include <vector>
#include "ray.cuh"
#include "device_random.cuh"
#include "device_texture.cuh"
#include "scene_entity.cuh"
#include "square.cuh"
#include "triangle.cuh"
#include "intersection.cuh"
#include "cuda_utils.cuh"
#include "device_fixed_vector.cuh"

#include <functional>

#define EPSILON 9.99999997475243E-07

enum Axis {
    X, Y, Z
};

constexpr size_t EntityGroupSize = 1500;

template<typename T>
struct TreeNode {
    Axis splitting_axis;
    float splitting_value;
    TreeNode *left;
    TreeNode *right;
    DeviceFixedVector<T> items;
};

enum class SplitComparison {
    GreaterOrEqual,
    Less,
    Both
};

class Scene {
public:
    Scene(std::vector<Square *> &walls, std::vector<Triangle *> &floors_ceilings,
          std::vector<SceneEntity *> &scene_entities, DeviceTexture *sky);

    __device__ bool intersect(const Ray &ray, Intersection &intersection, float tmax = FLT_MAX);

    [[nodiscard]] __device__ const DeviceTexture *sky() const { return m_sky; }

    void rebuild_entities(const std::vector<SceneEntity *> &scene_entities);

    DeviceFixedVector<SceneEntity*> m_emissive_entities;
    /*SceneEntity **m_emissive_entities;
    size_t m_emissive_entities_count;*/

    void add_light(SceneEntity *entity);
    void remove_light(SceneEntity *entity);

    void add_entity(SceneEntity *entity);

    void remove_entity(SceneEntity *entity);

private:
    TreeNode<Square *> *m_walls_root;
    TreeNode<Triangle *> *m_floors_ceilings_root;
    TreeNode<SceneEntity *> *m_entities_root;

    DeviceTexture *m_sky;

    __device__ bool intersect_walls(const Ray &ray, Intersection &intersection);

    __device__ bool intersect_floors_and_ceilings(const Ray &ray, Intersection &intersection);

    __device__ bool intersect_entities(const Ray &ray, Intersection &intersection);

    void add_entity(SceneEntity *entity, TreeNode<SceneEntity *> &node, Axis splitting_axis);

    void remove_entity(SceneEntity *entity, TreeNode<SceneEntity *> &node);

    template<typename T>
    void build_node(TreeNode<T> &node, std::vector<T> &items, Axis current_axis, bool valid_axes[3], size_t size_limit,
                    const std::function<void(std::vector<T> &items, Axis axis)> &sort_callback,
                    const std::function<glm::vec3(const T median)> &median_callback,
                    const std::function<SplitComparison(const T item, Axis axis,
                                                        float splitting_value)> &split_callback) {
        if (items.size() < size_limit) {
            node.items = DeviceFixedVector<T>(items, size_limit);
            node.left = nullptr;
            node.right = nullptr;
            return;
        }

        sort_callback(items, current_axis);

        auto half_size = items.size() / 2;
        auto median_point = median_callback(items[half_size]);
        auto splitting_value = median_point[static_cast<int>(current_axis)];

        std::vector<T> left_side;
        std::vector<T> right_side;

        left_side.reserve(half_size);
        right_side.reserve(half_size);

        for (auto item: items) {
            auto result = split_callback(item, current_axis, splitting_value);
            if (result == SplitComparison::GreaterOrEqual || result == SplitComparison::Both) {
                right_side.push_back(item);
            }

            if (result == SplitComparison::Less || result == SplitComparison::Both) {
                left_side.push_back(item);
            }
        }

        node.splitting_axis = current_axis;
        node.splitting_value = splitting_value;
        cuda_assert(cudaMallocManaged(&node.left, sizeof(TreeNode<T>)));
        cuda_assert(cudaMallocManaged(&node.right, sizeof(TreeNode<T>)));

        do {
            current_axis = static_cast<Axis>((current_axis + 1) % 3);
        } while (!valid_axes[current_axis]);

        build_node(*node.left, left_side, current_axis, valid_axes, size_limit, sort_callback, median_callback,
                   split_callback);
        build_node(*node.right, right_side, current_axis, valid_axes, size_limit, sort_callback, median_callback,
                   split_callback);
    }

    void build_walls_node(TreeNode<Square *> &node, std::vector<Square *> &walls, Axis current_axis, bool valid_axes[3],
                          size_t size_limit);

    static SplitComparison scene_entity_axis_comparison(const SceneEntity *item, Axis axis, float splitting_value) {
        auto sprite_max_bounds = item->sprite.calculate_max_bounds();
        auto min_bounds = item->position - glm::vec3(sprite_max_bounds.x / 2.0f, 0, sprite_max_bounds.x / 2.0f);
        auto max_bounds =
                item->position + glm::vec3(sprite_max_bounds.x / 2.0f, sprite_max_bounds.y, sprite_max_bounds.x / 2.0f);

        bool is_greater_or_equal = min_bounds[axis] >= splitting_value ||
                                   max_bounds[axis] >= splitting_value;

        bool is_less = min_bounds[axis] <= splitting_value ||
                       max_bounds[axis] <= splitting_value;

        if (is_greater_or_equal && is_less) {
            return SplitComparison::Both;
        } else if (is_greater_or_equal) {
            return SplitComparison::GreaterOrEqual;
        } else {
            return SplitComparison::Less;
        }
    };
};

#endif