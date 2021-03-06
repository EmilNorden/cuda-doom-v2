#include "scene.cuh"
#include "cuda_utils.cuh"
#include "device_stack.cuh"
#include <algorithm>


constexpr size_t EntityGroupSize = 200;
constexpr bool entity_valid_axes[3] = {true, false, true};

template<typename T>
struct NodeSearchData {
    TreeNode<T> *node;
    float tmin;
    float tmax;

    __device__ NodeSearchData()
            : node(nullptr), tmin(0), tmax(0) {

    }

    __device__ NodeSearchData(TreeNode<T> *n, float min, float max)
            : node(n), tmin(min), tmax(max) {

    }
};

Scene::Scene(std::vector<Square *> &walls, std::vector<Triangle *> &floors_ceilings, DeviceTexture *sky)
        : m_emissive_entities(200), m_sky(sky) {
    m_walls_root = create_device_type<TreeNode<Square *>>();
    m_floors_ceilings_root = create_device_type<TreeNode<Triangle *>>();
    m_entities_root = create_device_type<TreeNode<SceneEntity *>>();

    std::cout << "Building scene with " << walls.size() << " walls and" << floors_ceilings.size()
              << " floor/ceiling triangles\n";

    std::vector<LightInfo<Triangle *>> emissive_floors_ceilings;
    for (auto triangle: floors_ceilings) {
        if (triangle->material.has_emission()) {
            emissive_floors_ceilings.emplace_back(triangle, triangle->material.emission());
        }
    }
    m_emissive_floors_ceilings.reset(emissive_floors_ceilings);

    std::vector<LightInfo<Square *>> emissive_walls;
    for (auto square: walls) {
        if (square->material.has_emission()) {
            emissive_walls.emplace_back(square, square->material.emission());
        }
    }
    m_emissive_walls.reset(emissive_walls);


    bool valid_axes[3] = {true, false, true};
    auto walls_copy = walls;
    build_walls_node(*m_walls_root, walls_copy, Axis::X, valid_axes, 150);

    std::function<void(std::vector<Triangle *> &, Axis axis)> triangle_sort_callback = [](
            std::vector<Triangle *> &items,
            Axis axis) {
        std::sort(items.begin(), items.end(), [&](const Triangle *a, const Triangle *b) {
            auto a_mid_point = (a->v0 + a->v1 + a->v2) / 3.0f;
            auto b_mid_point = (b->v0 + b->v1 + b->v2) / 3.0f;

            return a_mid_point[axis] < b_mid_point[axis];
        });
    };

    std::function<glm::vec3(Triangle *median)> triangle_median_callback = [](const Triangle *median) {
        return median->v0;
    };

    std::function<SplitComparison(Triangle *item, Axis axis, float splitting_value)> triangle_split_callback = [](
            const Triangle *item, Axis axis, float splitting_value) {
        auto v0 = item->v0;
        auto v1 = item->v1;
        auto v2 = item->v2;

        bool is_greater_or_equal = v0[axis] >= splitting_value ||
                                   v1[axis] >= splitting_value ||
                                   v2[axis] >= splitting_value;

        bool is_less = v0[axis] < splitting_value ||
                       v1[axis] < splitting_value ||
                       v2[axis] < splitting_value;

        if (is_greater_or_equal && is_less) {
            return SplitComparison::Both;
        } else if (is_greater_or_equal) {
            return SplitComparison::GreaterOrEqual;
        } else {
            return SplitComparison::Less;
        }
    };

    auto floors_ceilings_copy = floors_ceilings;
    build_node(*m_floors_ceilings_root, floors_ceilings_copy, Axis::X, valid_axes, 150,
               triangle_sort_callback,
               triangle_median_callback,
               triangle_split_callback);
}

// TODO: Keep this here until i decide wether to keep the templated version. It's a mess.

void
Scene::build_walls_node(TreeNode<Square *> &node, std::vector<Square *> &walls, Axis current_axis, bool valid_axes[3],
                        size_t size_limit) {
    if (walls.size() < size_limit) {
        node.items = DeviceFixedVector<Square *>(walls, size_limit);
        node.left = nullptr;
        node.right = nullptr;
        return;
    }

    auto axis = static_cast<int>(current_axis);

    std::sort(walls.begin(), walls.end(), [&](const Square *a, const Square *b) {
        // Just compare top_left corner for now

        return a->top_left[axis] < b->top_left[axis];
    });

    auto half_size = walls.size() / 2;
    auto median_point = walls[half_size]->top_left;
    auto splitting_value = median_point[axis];

    std::vector<Square *> left_side;
    std::vector<Square *> right_side;

    left_side.reserve(half_size);
    right_side.reserve(half_size);

    for (auto &wall: walls) {
        auto v0 = wall->top_left;
        auto v1 = wall->top_left + (wall->horizontal_vec * wall->horizontal_len);
        auto v2 = wall->top_left + (wall->vertical_vec * wall->vertical_len);

        bool is_greater_or_equal = v0[axis] >= splitting_value ||
                                   v1[axis] >= splitting_value ||
                                   v2[axis] >= splitting_value;

        bool is_less = v0[axis] <= splitting_value ||
                       v1[axis] <= splitting_value ||
                       v2[axis] <= splitting_value;

        if (is_greater_or_equal) {
            right_side.push_back(wall);
        }
        if (is_less) {
            left_side.push_back(wall);
        }
    }

    node.splitting_axis = current_axis;
    node.splitting_value = splitting_value;
    cuda_assert(cudaMallocManaged(&node.left, sizeof(TreeNode<Square>)));
    cuda_assert(cudaMallocManaged(&node.right, sizeof(TreeNode<Square>)));

    do {
        current_axis = static_cast<Axis>((current_axis + 1) % 3);
    } while (!valid_axes[current_axis]);

    build_walls_node(*node.left, left_side, current_axis, valid_axes, size_limit);
    build_walls_node(*node.right, right_side, current_axis, valid_axes, size_limit);
}

template<typename T>
__device__ bool is_leaf(TreeNode<T> *node) {
    return node->items.data();
}

enum class RangePlaneComparison {
    BelowPlane,
    AbovePlane,
    BelowToAbove,
    AboveToBelow
};

template<typename T>
__device__ RangePlaneComparison
CompareRangeWithPlane(const Ray &ray, float tmin, float tmax, TreeNode<T> *node) {
    auto axis = (int) node->splitting_axis;
    // TODO: Extract components before performing multiplication etc.
    auto range_start = ray.origin()[axis] + (ray.direction()[axis] * tmin);
    auto range_end = ray.origin()[axis] + (ray.direction()[axis] * tmax);

    auto splitting_value = node->splitting_value;

    if (range_start < splitting_value && range_end < splitting_value) {
        return RangePlaneComparison::BelowPlane;
    } else if (range_start >= splitting_value && range_end >= splitting_value) {
        return RangePlaneComparison::AbovePlane;
    } else if (range_start < splitting_value && range_end >= splitting_value) {
        return RangePlaneComparison::BelowToAbove;
    } else if (range_start >= splitting_value && range_end < splitting_value) {
        return RangePlaneComparison::AboveToBelow;
    }

    assert(false);
}


template<bool AnyHit>
__device__ bool
intersects_walls_node(const Ray &ray, TreeNode<Square *> *node, Intersection &intersection, float tmax) {
    auto success = false;
    for (auto i = 0; i < node->items.count(); ++i) {
        float hit_distance = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        glm::vec3 normal{};
        if (intersects_wall(ray, node->items[i], hit_distance, u, v, normal) && hit_distance < tmax) {
            if (node->items[i]->material.sample_diffuse({u, v}) > 0xFF) {
                continue;
            }
            if (AnyHit) {
                return true;
            }
            tmax = hit_distance;
            intersection.distance = hit_distance;
            intersection.u = u;
            intersection.v = v;
            intersection.material = &node->items[i]->material;
            intersection.world_normal = normal;

            success = true;
        }
    }

    return success;
}

template <bool AnyHit>
__device__ bool
intersects_floors_and_ceilings_node(const Ray &ray, TreeNode<Triangle *> *node, Intersection &intersection,
                                    float tmax) {
    auto success = false;
    for (auto i = 0; i < node->items.count(); ++i) {
        float hit_distance = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (intersects_triangle(ray, node->items[i], hit_distance, u, v) && hit_distance < tmax) {
            if(AnyHit) {
                return true;
            }
            tmax = hit_distance;
            intersection.distance = hit_distance;
            intersection.u = u;
            intersection.v = v;
            intersection.material = &node->items[i]->material;
            if (ray.direction().y >= 0.0f) {
                // Going upwards, must have hit ceiling
                intersection.world_normal = glm::vec3{0.0f, -1.0f, 0.0f};
            } else {
                // Going downwards, must have hit floor
                intersection.world_normal = glm::vec3{0.0f, 1.0f, 0.0f};
            }
            success = true;
        }
    }

    return success;
}

template <bool AnyHit>
__device__ bool Scene::intersect_walls(const Ray &ray, Intersection &intersection) {
    constexpr int StackSize = 20;

    DeviceStack<StackSize, NodeSearchData<Square *>> nodes;

    nodes.push({
                       m_walls_root,
                       FLT_MIN,
                       intersection.distance
               });

    while (!nodes.is_empty()) {
        auto current = nodes.pop();
        auto node = current.node;
        auto tmin = current.tmin;
        auto tmax = current.tmax;

        if (is_leaf(node)) {
            if (intersects_walls_node<AnyHit>(ray, node, intersection, tmax)) {
                return true;
            }
        } else {
            auto a = (int) node->splitting_axis;
            auto thit = (node->splitting_value - ray.origin()[a]) * ray.inverse_direction()[a];

            switch (CompareRangeWithPlane(ray, tmin, tmax, node)) {
                case RangePlaneComparison::AbovePlane:
                    nodes.push(NodeSearchData{node->right, tmin, tmax});
                    break;
                case RangePlaneComparison::BelowPlane:
                    nodes.push(NodeSearchData{node->left, tmin, tmax});
                    break;
                case RangePlaneComparison::AboveToBelow:
                    nodes.push(NodeSearchData{node->left, thit, tmax});
                    nodes.push(NodeSearchData{node->right, tmin, thit});

                    break;
                case RangePlaneComparison::BelowToAbove:
                    nodes.push(NodeSearchData{node->right, thit, tmax});
                    nodes.push(NodeSearchData{node->left, tmin, thit});
                    break;
            }
        }
    }

    return false;
}

template <bool AnyHit>
__device__ bool Scene::intersect_floors_and_ceilings(const Ray &ray, Intersection &intersection) {
    constexpr int StackSize = 20;

    DeviceStack<StackSize, NodeSearchData<Triangle *>> nodes;

    nodes.push({
                       m_floors_ceilings_root,
                       FLT_MIN,
                       intersection.distance
               });

    while (!nodes.is_empty()) {
        auto current = nodes.pop();
        auto node = current.node;
        auto tmin = current.tmin;
        auto tmax = current.tmax;

        if (is_leaf(node)) {
            if (intersects_floors_and_ceilings_node<AnyHit>(ray, node, intersection, tmax)) {
                return true;
            }
        } else {
            auto a = (int) node->splitting_axis;
            auto thit = (node->splitting_value - ray.origin()[a]) * ray.inverse_direction()[a];

            switch (CompareRangeWithPlane(ray, tmin, tmax, node)) {
                case RangePlaneComparison::AbovePlane:
                    nodes.push(NodeSearchData{node->right, tmin, tmax});
                    break;
                case RangePlaneComparison::BelowPlane:
                    nodes.push(NodeSearchData{node->left, tmin, tmax});
                    break;
                case RangePlaneComparison::AboveToBelow:
                    nodes.push(NodeSearchData{node->left, thit, tmax});
                    nodes.push(NodeSearchData{node->right, tmin, thit});

                    break;
                case RangePlaneComparison::BelowToAbove:
                    nodes.push(NodeSearchData{node->right, thit, tmax});
                    nodes.push(NodeSearchData{node->left, tmin, thit});
                    break;
            }
        }
    }

    return false;
}

template <bool AnyHit>
__device__ bool
intersects_entity_node(const Ray &ray, TreeNode<SceneEntity *> *node, Intersection &intersection, float tmax,
                       bool ignore_player) {
    auto success = false;
    for (auto i = 0; i < node->items.count(); ++i) {
        float hit_distance = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (intersects_scene_entity(ray, node->items[i], hit_distance, u, v) && hit_distance < tmax) {
            if (ignore_player && node->items[i]->is_player) {
                continue;
            }

            auto material = node->items[i]->sprite.get_material(node->items[i]->frame, node->items[i]->rotation);

            if (material->sample_diffuse({u, v}) > 0xFF) {
                continue;
            }

            if(AnyHit) {
                return true;
            }

            tmax = hit_distance;
            intersection.u = u;
            intersection.v = v;
            intersection.material = material;
            intersection.world_normal = ray.direction() * -1.0f; // TODO Dirty workaround for now
            intersection.distance = hit_distance;
            success = true;
        }
    }

    return success;
}

template <bool AnyHit>
__device__ bool Scene::intersect_entities(const Ray &ray, Intersection &intersection, bool ignore_player) {
    constexpr int StackSize = 20;

    DeviceStack<StackSize, NodeSearchData<SceneEntity *>> nodes;

    nodes.push({
                       m_entities_root,
                       FLT_MIN,
                       intersection.distance
               });

    while (!nodes.is_empty()) {
        auto current = nodes.pop();
        auto node = current.node;
        auto tmin = current.tmin;
        auto tmax = current.tmax;

        if (is_leaf(node)) {
            if (intersects_entity_node<AnyHit>(ray, node, intersection, tmax, ignore_player)) {
                return true;
            }
        } else {
            auto a = (int) node->splitting_axis;
            auto thit = (node->splitting_value - ray.origin()[a]) * ray.inverse_direction()[a];

            switch (CompareRangeWithPlane(ray, tmin, tmax, node)) {
                case RangePlaneComparison::AbovePlane:
                    nodes.push(NodeSearchData{node->right, tmin, tmax});
                    break;
                case RangePlaneComparison::BelowPlane:
                    nodes.push(NodeSearchData{node->left, tmin, tmax});
                    break;
                case RangePlaneComparison::AboveToBelow:
                    nodes.push(NodeSearchData{node->left, thit, tmax});
                    nodes.push(NodeSearchData{node->right, tmin, thit});

                    break;
                case RangePlaneComparison::BelowToAbove:
                    nodes.push(NodeSearchData{node->right, thit, tmax});
                    nodes.push(NodeSearchData{node->left, tmin, thit});
                    break;
            }
        }
    }

    return false;
}

__device__ bool Scene::intersect(const Ray &ray, Intersection &intersection, bool ignore_player, float tmax) {
    intersection.distance = tmax;

    auto intersects_walls = intersect_walls<false>(ray, intersection);
    auto intersects_floors_or_ceilings = intersect_floors_and_ceilings<false>(ray, intersection);
    auto intersects_entities = intersect_entities<false>(ray, intersection, ignore_player);
    //return intersects_floors_or_ceilings;
    return intersects_walls || intersects_floors_or_ceilings || intersects_entities;
}


__device__ bool Scene::intersect_walls_any(const Ray &ray, float tmax) {
    Intersection intersection;
    intersection.distance = tmax;
    return intersect_walls<true>(ray, intersection);
}

__device__ bool Scene::intersect_floors_and_ceilings_any(const Ray &ray, float tmax) {
    Intersection intersection;
    intersection.distance = tmax;
    return intersect_floors_and_ceilings<true>(ray, intersection);
}

__device__ bool Scene::intersect_entities_any(const Ray &ray, float tmax) {
    Intersection intersection;
    intersection.distance = tmax;
    return intersect_entities<true>(ray, intersection, false);
}

__device__ bool Scene::intersect_any(const Ray &ray, float tmax) {
    return intersect_walls_any(ray, tmax) ||
           intersect_floors_and_ceilings_any(ray, tmax) ||
           intersect_entities_any(ray, tmax);
}

void Scene::add_light(SceneEntity *entity) {
    if (m_emissive_entities.is_full()) {
        return;
    }

    m_emissive_entities.push_back(LightInfo(entity, entity->sprite.calculate_max_emission()));
}

void Scene::remove_light(SceneEntity *entity) {
    m_emissive_entities.remove_by([&](LightInfo<SceneEntity *> &item) {
        return item.geometry() == entity;
    });
}

void Scene::add_entities(const std::vector<SceneEntity *> &entities) {
    for (auto entity: entities) {
        if (entity->sprite.has_emissive_frames()) {
            add_light(entity);
        }
    }

    std::function<void(std::vector<SceneEntity *> &, Axis axis)> entity_sort_callback = [](
            std::vector<SceneEntity *> &items,
            Axis axis) {
        std::sort(items.begin(), items.end(), [&](const SceneEntity *a, const SceneEntity *b) {
            return a->position[static_cast<int>(axis)] < b->position[static_cast<int>(axis)];
        });
    };

    std::function<glm::vec3(SceneEntity *median)> entity_median_callback = [](SceneEntity *median) {
        return median->position;
    };

    std::function<SplitComparison(SceneEntity *item, Axis axis, float splitting_value)> entity_split_callback = [](
            const SceneEntity *entity, Axis axis, float split_value) {
        return Scene::scene_entity_axis_comparison(entity, axis, split_value);
    };

    auto entities_copy = entities;
    bool valid_axes[3] = {entity_valid_axes[0], entity_valid_axes[1], entity_valid_axes[2]};
    build_node(*m_entities_root, entities_copy, Axis::X, valid_axes, EntityGroupSize,
               entity_sort_callback,
               entity_median_callback,
               entity_split_callback);
}

void Scene::add_entity(SceneEntity *entity) {
    if (entity->sprite.has_emissive_frames()) {
        add_light(entity);
    }
    add_entity(entity, *m_entities_root);
}

void Scene::refresh_entity(SceneEntity *entity) {
    if (remove_entity(entity, *m_entities_root)) {
        // If we managed to remove it from the scene, we can safely add it again.
        // If we didnt manage to remove it, it means it has already been removed (ie explosions or smoke puffs that live for a short time)
        // Adding it again would mean we'd have frozen explosions around the map
        add_entity(entity, *m_entities_root);
    }
}

void Scene::add_entity(SceneEntity *entity, TreeNode<SceneEntity *> &node) {
    if (node.items.data()) {
        if (!node.items.is_full()) {
            node.items.push_back(entity);
            return;
        }
        printf("ITS SPLITTING TIME\n");

        std::sort(node.items.data(), node.items.data() + node.items.count(),
                  [&](const SceneEntity *a, const SceneEntity *b) {
                      return a->position[static_cast<int>(node.splitting_axis)] <
                             b->position[static_cast<int>(node.splitting_axis)];
                  });

        auto half_size = node.items.count() / 2;
        auto median_point = node.items[half_size]->position;
        auto splitting_value = median_point[static_cast<int>(node.splitting_axis)];

        std::vector<SceneEntity *> left_side;
        std::vector<SceneEntity *> right_side;

        left_side.reserve(half_size);
        right_side.reserve(half_size);

        for (int i = 0; i < node.items.count(); ++i) {
            auto result = scene_entity_axis_comparison(node.items[i], node.splitting_axis, splitting_value);
            if (result == SplitComparison::GreaterOrEqual || result == SplitComparison::Both) {
                right_side.push_back(node.items[i]);
            }

            if (result == SplitComparison::Less || result == SplitComparison::Both) {
                left_side.push_back(node.items[i]);
            }
        }


        Axis next_axis = node.splitting_axis;
        do {
            next_axis = static_cast<Axis>((next_axis + 1) % 3);
        } while (!entity_valid_axes[next_axis]);

        node.items.reset();
        node.splitting_value = splitting_value;
        cuda_assert(cudaMallocManaged(&node.left, sizeof(TreeNode<SceneEntity *>)));
        cuda_assert(cudaMallocManaged(&node.right, sizeof(TreeNode<SceneEntity *>)));
        node.left->items = DeviceFixedVector<SceneEntity *>(left_side, EntityGroupSize);
        node.left->splitting_axis = next_axis;
        node.right->items = DeviceFixedVector<SceneEntity *>(right_side, EntityGroupSize);
        node.right->splitting_axis = next_axis;
    }

    auto axis_comparison = scene_entity_axis_comparison(entity, node.splitting_axis, node.splitting_value);
    switch (axis_comparison) {
        case SplitComparison::GreaterOrEqual:
            add_entity(entity, *node.right);
            break;
        case SplitComparison::Less:
            add_entity(entity, *node.left);
            break;
        case SplitComparison::Both:
            add_entity(entity, *node.right);
            add_entity(entity, *node.left);
            break;
    }
}

bool Scene::remove_entity(SceneEntity *entity) {
    if (entity->sprite.has_emissive_frames()) {
        remove_light(entity);
    }
    return remove_entity(entity, *m_entities_root);
}

bool Scene::remove_entity(SceneEntity *entity, TreeNode<SceneEntity *> &node) {
    if (node.items.data()) {
        return node.items.remove(entity);
    }

    auto axis_comparison = scene_entity_axis_comparison(entity, node.splitting_axis, node.splitting_value);
    switch (axis_comparison) {
        case SplitComparison::GreaterOrEqual:
            return remove_entity(entity, *node.right);
        case SplitComparison::Less:
            return remove_entity(entity, *node.left);
        case SplitComparison::Both:
            auto removed_from_right = remove_entity(entity, *node.right);
            auto removed_from_left = remove_entity(entity, *node.left);
            return removed_from_right || removed_from_left;
    }
}