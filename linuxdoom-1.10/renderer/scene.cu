#include "scene.cuh"
#include "cuda_utils.cuh"
#include "device_stack.cuh"
#include <algorithm>

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

Scene::Scene(std::vector<Square*> &walls, std::vector<Triangle*> &floors_ceilings, std::vector<MapThing*> &map_things) {
    m_walls_root = create_device_type<TreeNode<Square*>>();
    m_floors_ceilings_root = create_device_type<TreeNode<Triangle*>>();
    m_map_things_root = create_device_type<TreeNode<MapThing*>>();

    std::cout << "Building scene with " << walls.size() << " walls, " << floors_ceilings.size()
              << " floor/ceiling triangles and " << map_things.size() << " things\n";


    bool valid_axes[3] = {true, false, true};
    auto walls_copy = walls;
    build_walls_node(*m_walls_root, walls_copy, Axis::X, valid_axes, 150);

    std::function<void(std::vector<Triangle*> &, Axis axis)> triangle_sort_callback = [](std::vector<Triangle*> &items,
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

    std::function<void(std::vector<MapThing*> &, Axis axis)> things_sort_callback = [](std::vector<MapThing*> &items,
                                                                                       Axis axis) {
        std::sort(items.begin(), items.end(), [&](const MapThing *a, const MapThing *b) {
            return a->position[static_cast<int>(axis)] < b->position[static_cast<int>(axis)];
        });
    };

    std::function<glm::vec3(MapThing *median)> thing_median_callback = [](MapThing *median) {
        return median->position;
    };

    std::function<SplitComparison(MapThing *item, Axis axis, float splitting_value)> thing_split_callback = [](
            MapThing *item, Axis axis, float splitting_value) {
        auto min_bounds = item->position - glm::vec3(item->max_size.x / 2.0f, 0, item->max_size.x / 2.0f);
        auto max_bounds = item->position + glm::vec3(item->max_size.x / 2.0f, item->max_size.y, item->max_size.x / 2.0f);

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

    auto map_things_copy = map_things;
    build_node(*m_map_things_root, map_things_copy, Axis::X, valid_axes, 2000,
               things_sort_callback,
               thing_median_callback,
               thing_split_callback);
}

// TODO: Keep this here until i decide wether to keep the templated version. It's a mess.

void Scene::build_walls_node(TreeNode<Square*> &node, std::vector<Square*> &walls, Axis current_axis, bool valid_axes[3], size_t size_limit) {
    if (walls.size() < size_limit) {
        cuda_assert(cudaMalloc(&node.items, sizeof(Square*) * walls.size()));
        cuda_assert(cudaMemcpy(node.items, walls.data(), sizeof(Square*) * walls.size(), cudaMemcpyHostToDevice));
        node.item_count = walls.size();
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

    std::vector<Square*> left_side;
    std::vector<Square*> right_side;

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
        if(is_less) {
            left_side.push_back(wall);
        }
    }

    node.splitting_axis = current_axis;
    node.splitting_value = splitting_value;
    node.items = nullptr;
    node.item_count = 0;
    cuda_assert(cudaMallocManaged(&node.left, sizeof(TreeNode<Square>)));
    cuda_assert(cudaMallocManaged(&node.right, sizeof(TreeNode<Square>)));

    do {
        current_axis = static_cast<Axis>((current_axis + 1) % 3);
    } while(!valid_axes[current_axis]);

    build_walls_node(*node.left, left_side, current_axis, valid_axes, size_limit);
    build_walls_node(*node.right, right_side, current_axis, valid_axes, size_limit);
}

template<typename T>
__device__ bool is_leaf(TreeNode<T> *node) {
    return node->items;
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


__device__ bool intersects_walls_node(const Ray &ray, TreeNode<Square*> *node, Intersection &intersection, float tmax) {
    auto success = false;
    for (auto i = 0; i < node->item_count; ++i) {
        float hit_distance = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        glm::vec3 normal{};
        if (intersects_wall(ray, node->items[i], hit_distance, u, v, normal) && hit_distance < tmax) {
            if (node->items[i]->texture->sample({u, v}) > 0xFF) {
                continue;
            }
            tmax = hit_distance;
            intersection.distance = hit_distance;
            intersection.u = u;
            intersection.v = v;
            intersection.texture = node->items[i]->texture;
            intersection.world_normal = normal;
            success = true;
        }
    }

    return success;
}

__device__ bool
intersects_floors_and_ceilings_node(const Ray &ray, TreeNode<Triangle*> *node, Intersection &intersection, float tmax) {
    auto success = false;
    for (auto i = 0; i < node->item_count; ++i) {
        float hit_distance = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (intersects_triangle(ray, node->items[i], hit_distance, u, v) && hit_distance < tmax) {
            tmax = hit_distance;
            intersection.distance = hit_distance;
            intersection.u = u;
            intersection.v = v;
            intersection.texture = node->items[i]->texture;
            if(ray.direction().y >= 0.0f) {
                // Going upwards, must have hit ceiling
                intersection.world_normal = glm::vec3{0.0f, -1.0f, 0.0f};
            }
            else {
                // Going downwards, must have hit floor
                intersection.world_normal = glm::vec3{0.0f, -1.0f, 0.0f};
            }
            success = true;
        }
    }

    return success;
}

__device__ bool Scene::intersect_walls(const Ray &ray, Intersection &intersection) {
    constexpr int StackSize = 20;

    DeviceStack<StackSize, NodeSearchData<Square*>> nodes;

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
            if (intersects_walls_node(ray, node, intersection, tmax)) {
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

__device__ bool Scene::intersect_floors_and_ceilings(const Ray &ray, Intersection &intersection) {
    constexpr int StackSize = 20;

    DeviceStack<StackSize, NodeSearchData<Triangle*>> nodes;

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
            if (intersects_floors_and_ceilings_node(ray, node, intersection, tmax)) {
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

__device__ bool intersects_map_things_node(const Ray& ray, TreeNode<MapThing*> *node, Intersection &intersection, float tmax) {
    auto success = false;
    for (auto i = 0; i < node->item_count; ++i) {
        float hit_distance = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (intersects_map_thing(ray, node->items[i], hit_distance, u, v) && hit_distance < tmax) {
            auto texture = node->items[i]->sprite.get_texture(node->items[i]->frame,node->items[i]->rotation);
            if (texture->sample({u, v}) > 0xFF) {
                continue;
            }
            tmax = hit_distance;
            intersection.u = u;
            intersection.v = v;
            intersection.texture = texture;
            intersection.world_normal = glm::vec3(0,1,0); // TODO Dirty workaround for now
            intersection.distance = hit_distance;
            success = true;
        }
    }

    return success;
}

__device__ bool Scene::intersect_things(const Ray &ray, Intersection &intersection) {
    constexpr int StackSize = 20;

    DeviceStack<StackSize, NodeSearchData<MapThing*>> nodes;

    nodes.push({
                       m_map_things_root,
                       FLT_MIN,
                       intersection.distance
               });

    while (!nodes.is_empty()) {
        auto current = nodes.pop();
        auto node = current.node;
        auto tmin = current.tmin;
        auto tmax = current.tmax;

        if (is_leaf(node)) {
            if (intersects_map_things_node(ray, node, intersection, tmax)) {
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

__device__ bool Scene::intersect(const Ray &ray, Intersection &intersection) {
    intersection.distance = FLT_MAX;

    auto intersects_walls = intersect_walls(ray, intersection);
    auto intersects_floors_or_ceilings = intersect_floors_and_ceilings(ray, intersection);
    //auto intersects_things = intersect_things(ray, intersection);

    //return intersects_walls || intersects_floors_or_ceilings || intersects_things;
    return intersects_walls ||  intersects_floors_or_ceilings;
}
