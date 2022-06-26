#include "rt_raytracing.cuh"
#include "r_state.h"
#include "z_zone.h" // FOR PU_CACHE
#include "w_wad.h"
#include "renderer/device_texture.cuh"
#include "renderer/cuda_utils.cuh"
#include "wad/graphics_data.cuh"
#include "renderer/scene.cuh"
#include "doomstat.h"
#include "r_sky.h"
#include "rt_raytracing_scene.cuh"
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
#include <set>
// Below are only for debugging polygons
#include <FreeImage.h>
#include <fmt/core.h>

class TextureCache {
public:
    DeviceTexture *get_texture_for_flat(short flatnum) {
        auto it = m_flat_textures.find(flatnum);
        if (it != m_flat_textures.end()) {
            return it->second;
        }
        return nullptr;
    }

    DeviceTexture *get_texture(short texture_number) {
        auto it = m_textures.find(texture_number);
        if (it != m_textures.end()) {
            return it->second;
        }
        return nullptr;
    }

    void insert_flat_texture(short flatnum, DeviceTexture *texture) {
        m_flat_textures[flatnum] = texture;
    }

    void insert_texture(short texture_number, DeviceTexture *texture) {
        m_textures[texture_number] = texture;
    }

private:
    std::unordered_map<short, DeviceTexture *> m_flat_textures;
    std::unordered_map<short, DeviceTexture *> m_textures;

};

struct Side {
    Square *top;
    Square *middle;
    Square *bottom;
};

namespace detail {
    TextureCache texture_cache;
    std::unordered_map<side_t *, Side> sidedef_to_wall_lookup;
}

struct SceneData {
    std::vector<Square *> walls;
    std::vector<Triangle *> triangles;
    std::vector<SceneEntity *> entities;
    std::vector<DeviceTexture *> device_textures;
    std::unordered_map<sector_t *, SectorGeometry> sector_geometry;
};

typedef std::vector<glm::vec2> Polygon;

void debug_polygon(Polygon &polygon, const std::string& name);

bool on_segment(glm::vec2 p, glm::vec2 q, glm::vec2 r);

int orientation(glm::vec2 p, glm::vec2 q, glm::vec2 r);

bool lines_intersect(glm::vec2 p1, glm::vec2 q1, glm::vec2 p2, glm::vec2 q2);

bool is_point_inside_polygon(const Polygon &container, const glm::vec2 &point);

bool is_polygon_inside_other(const Polygon &container, const Polygon &test_polygon);

void combine_polygons(std::vector<Polygon> &polygons, size_t parent_polygon_index, size_t child_polygon_index, std::vector<bool>& parent_polygon_weld_points);

bool is_polygon_cw_winding(const std::vector<glm::vec2> &polygon);

void create_mesh_from_polygon(
        sector_t *sector,
        SceneData &scene_data,
        std::vector<glm::vec2> &polygon);

DeviceTexture *get_device_texture(short texture_number,
                                  wad::Wad &wad,
                                  wad::GraphicsData &graphics_data);

DeviceTexture *get_device_texture_from_flat(short flatnum);

DeviceTexture *create_device_texture_from_flat(intptr_t lump_number);

Square *create_sector_adjacent_wall(short texture_number,
                                    wad::Wad &wad, wad::GraphicsData &graphics_data,
                                    int front_sector_height, int back_sector_height,
                                    vertex_t *start_vertex,
                                    vertex_t *end_vertex);

Square *create_main_wall(short texture_number, wad::Wad &wad, wad::GraphicsData &graphics_data,
                         int floor_height, int ceiling_height, vertex_t *start_vertex, vertex_t *end_vertex,
                         int line_flags);

BuildSceneResult RT_BuildScene(wad::Wad &wad, wad::GraphicsData &graphics_data) {
    SceneData scene_data;
    FreeImage_Initialise();
    sector_t *sector_ptr = sectors;
    for (int sector_number = 0; sector_number < numsectors; sector_number++, sector_ptr++) {


        std::vector<int> all_sides;
        side_t *side_ptr = sides;
        for (int j = 0; j < numsides; j++, side_ptr++) {
            if (side_ptr->sector == sector_ptr) {
                all_sides.push_back(j);
            }
        }

        std::vector<line_t> all_lines;
        line_t *line_ptr = lines;
        for (int j = 0; j < numlines; j++, line_ptr++) {
            for (auto side: all_sides) {
                if (line_ptr->sidenum[0] == side || line_ptr->sidenum[1] == side) { // 0 is right?
                    all_lines.push_back(*line_ptr);
                }
            }
        }

        // FIND ALL POLYGONS
        auto poly_count = 0;
        std::vector<Polygon> polygons;
        while (!all_lines.empty()) {
            std::vector<line_t> sorted_lines;
            sorted_lines.push_back(all_lines.back());
            all_lines.pop_back();
            auto start_vertex = sorted_lines[0].v1;

            while (true) {
                auto &current = sorted_lines.back();

                if (current.v2 == start_vertex) {
                    break;
                }

                auto foo = std::find_if(all_lines.begin(), all_lines.end(), [&](const line_t &linedef) {
                    return linedef.v1 == current.v2 || linedef.v2 == current.v2;
                });

                if (foo == all_lines.end()) {
                    break;
                }

                auto next_line = *foo;
                if (next_line.v2 == current.v2) {
                    std::swap(next_line.v1, next_line.v2);
                }

                sorted_lines.push_back(next_line);
                all_lines.erase(foo);
            }

            Polygon polygon;
            for (auto &line: sorted_lines) {
                polygon.emplace_back(glm::vec2{RT_FixedToFloating(line.v1->x), RT_FixedToFloating(line.v1->y)});
            }
            polygons.push_back(polygon);
        }

        // BUILD A MAP OF PARENT-CHILD RELATIONSHIPS
        std::unordered_multimap<size_t, size_t> parent_map;
        for (int i = 0; i < polygons.size(); ++i) {
            auto &current_polygon = polygons[i];

            for (int j = 0; j < polygons.size(); ++j) {
                if (i == j) {
                    continue;
                }

                if (is_polygon_inside_other(current_polygon, polygons[j])) {
                    // 'i' is the parent of 'j'
                    parent_map.insert({i, j});
                }
            }
        }

        auto is_direct_ancestor = [](size_t parent_polygon, size_t child,
                                     std::unordered_multimap<size_t, size_t> &parent_map) {
            auto children = parent_map.equal_range(parent_polygon);
            for (auto it = children.first; it != children.second; ++it) {
                if (it->second == child) {
                    continue;
                }

                // TODO: Make into a real function and make recursive call to handle long chains of ancestors
                auto grand_children = parent_map.equal_range(it->second);
                for (auto grand_it = grand_children.first; grand_it != grand_children.second; ++grand_it) {
                    if (grand_it->second == child) {
                        return false;
                    }
                }
            }

            return true;
        };

        auto is_childless = [](size_t polygon, std::unordered_multimap<size_t, size_t> &parent_map) {
            auto children = parent_map.equal_range(polygon);
            return children.first == children.second;
        };

        std::vector<bool> polygon_is_pruned(polygons.size(), false);
        std::vector<std::vector<bool>> polygon_vertices_weld_points;
        polygon_vertices_weld_points.resize(polygons.size());
        for(int i = 0; i < polygons.size(); ++i) {
            polygon_vertices_weld_points[i] = std::vector<bool>(polygons[i].size(), false);
        }

        // FLATTEN PARENT-CHILDS INTO SINGLE POLYGONS
        while (true) {
            auto has_pruned_polygon = false;
            //for (int i = polygons.size() - 1; i >= 0; --i) {
            for (int i = 0; i < polygons.size(); ++i) {
                if (polygon_is_pruned[i]) {
                    continue;
                }

                auto children = parent_map.equal_range(i);
                for (auto it = children.first; it != children.second; ++it) {
                    if (is_direct_ancestor(i, it->second, parent_map) && is_childless(it->second, parent_map)) {
                        if(sector_number == 88) {
                            auto foo = 334;
                        }
                        combine_polygons(polygons, i, it->second, polygon_vertices_weld_points[i]);
                        if(sector_number == 88) {
                            debug_polygon(polygons[i], fmt::format("combine_{}_{}_{}", sector_number, i, it->second));
                        }
                        polygon_is_pruned[it->second] = true;
                        it = parent_map.erase(it);
                        has_pruned_polygon = true;

                        if (it == children.second) { break; }
                    }
                }
            }

            if (!has_pruned_polygon) {
                break;
            }
        }

        for (int i = 0; i < polygons.size(); ++i) {
            if (polygon_is_pruned[i]) {
                continue;
            }

            //debug_polygon(polygons[i], fmt::format("{}_{}", sector_number, i));

            create_mesh_from_polygon(
                    sector_ptr,
                    scene_data,
                    polygons[i]);
        }

    }

    auto line_ptr = lines;
    for (int i = 0; i < numlines; i++, line_ptr++) {

        auto start_vertex = line_ptr->v1;
        auto end_vertex = line_ptr->v2;

        // REminder: line_ptr->sidenum[1] is left side
        if (line_ptr->sidenum[0] > -1 && line_ptr->sidenum[1] > -1) {
            auto left_side = &sides[line_ptr->sidenum[1]];
            auto right_side = &sides[line_ptr->sidenum[0]];

            auto left_sector = left_side->sector;
            auto right_sector = right_side->sector;

            auto left_tag = left_sector->tag;
            auto right_tag = right_sector->tag;
            if(left_tag == 14 || right_tag == 14) {
                int dffdfg=43;
            }

            auto left_upper_wall = create_sector_adjacent_wall(
                    left_side->toptexture,
                    wad,
                    graphics_data,
                    left_sector->ceilingheight,
                    right_sector->ceilingheight,
                    start_vertex,
                    end_vertex);

            if (left_upper_wall) {
                // if (line.special_type == 117) { // Vertical door
                scene_data.sector_geometry[left_side->sector].top_walls.push_back({left_upper_wall,
                                                                                   RT_FixedToFloating(
                                                                                           right_sector->ceilingheight)});


                scene_data.sector_geometry[right_side->sector].adjacent_top_walls.push_back(
                        {left_upper_wall, RT_FixedToFloating(left_sector->ceilingheight)});

                scene_data.walls.push_back(left_upper_wall);
                detail::sidedef_to_wall_lookup[left_side].top = left_upper_wall;
            }

            auto right_upper_wall = create_sector_adjacent_wall(
                    right_side->toptexture,
                    wad,
                    graphics_data,
                    left_sector->ceilingheight,
                    right_sector->ceilingheight,
                    start_vertex,
                    end_vertex);

            if (right_upper_wall) {
                //if (line.special_type == 117) { // Vertical door
                scene_data.sector_geometry[left_side->sector].top_walls.push_back({right_upper_wall,
                                                                                   RT_FixedToFloating(
                                                                                           right_sector->ceilingheight)});


                scene_data.sector_geometry[right_side->sector].adjacent_top_walls.push_back({right_upper_wall,
                                                                                    RT_FixedToFloating(
                                                                                            left_sector->ceilingheight)});


                scene_data.walls.push_back(right_upper_wall);
                detail::sidedef_to_wall_lookup[right_side].top = right_upper_wall;
            }

            auto left_lower_wall = create_sector_adjacent_wall(
                    left_side->bottomtexture,
                    wad,
                    graphics_data,
                    left_sector->floorheight,
                    right_sector->floorheight,
                    start_vertex,
                    end_vertex);

            if (left_lower_wall) {
                //if (line.special_type == 117) { // Vertical door
                scene_data.sector_geometry[left_side->sector].bottom_walls.push_back({left_lower_wall,
                                                                                      RT_FixedToFloating(
                                                                                              right_sector->floorheight)});


                scene_data.sector_geometry[right_side->sector].adjacent_bottom_walls.push_back({left_lower_wall,
                                                                                                RT_FixedToFloating(
                                                                                                        left_sector->floorheight)});

                scene_data.walls.push_back(left_lower_wall);
                detail::sidedef_to_wall_lookup[left_side].bottom = left_lower_wall;
            }

            auto right_lower_wall = create_sector_adjacent_wall(
                    right_side->bottomtexture,
                    wad,
                    graphics_data,
                    left_sector->floorheight,
                    right_sector->floorheight,
                    start_vertex,
                    end_vertex);

            if (right_lower_wall) {
                //if (line.special_type == 117) { // Vertical door
                scene_data.sector_geometry[left_side->sector].bottom_walls.push_back({right_lower_wall,
                                                                                      RT_FixedToFloating(
                                                                                              right_sector->floorheight)});


                scene_data.sector_geometry[right_side->sector].adjacent_bottom_walls.push_back({right_lower_wall,
                                                                                                RT_FixedToFloating(
                                                                                                        left_sector->floorheight)});

                scene_data.walls.push_back(right_lower_wall);
                detail::sidedef_to_wall_lookup[right_side].bottom = right_lower_wall;
            }

        }

        // left side
        if (line_ptr->sidenum[1] > -1) {
            auto side = &sides[line_ptr->sidenum[1]];
            auto sector = side->sector;
            if(sector->tag == 14) {
                int sdfsdf = 43;
            }

            auto wall = create_main_wall(
                    side->midtexture,
                    wad,
                    graphics_data,
                    sector->floorheight,
                    sector->ceilingheight,
                    start_vertex,
                    end_vertex,
                    line_ptr->flags);

            if (wall) {
                //if (line.special_type == 117) { // Vertical door
                scene_data.sector_geometry[side->sector].middle_walls.push_back(wall);

                scene_data.walls.push_back(wall);
                detail::sidedef_to_wall_lookup[side].middle = wall;
            }
        }

        if (line_ptr->sidenum[0] > -1) {
            auto side = &sides[line_ptr->sidenum[0]];
            auto sector = side->sector;

            if(sector->tag == 14) {
                int fgfgf=43;
            }

            auto wall = create_main_wall(
                    side->midtexture,
                    wad,
                    graphics_data,
                    sector->floorheight,
                    sector->ceilingheight,
                    start_vertex,
                    end_vertex,
                    line_ptr->flags);

            if (wall) {
                //if (line.special_type == 117) { // Vertical door
                scene_data.sector_geometry[side->sector].middle_walls.push_back(wall);

                scene_data.walls.push_back(wall);
                detail::sidedef_to_wall_lookup[side].middle = wall;
            }
        }
    }

    auto sky_texture = get_device_texture(skytexture, wad, graphics_data);

    return BuildSceneResult{
            create_device_type<Scene>(scene_data.walls, scene_data.triangles, scene_data.entities, sky_texture),
            scene_data.sector_geometry
    };
}

Square *create_main_wall(short texture_number, wad::Wad &wad, wad::GraphicsData &graphics_data,
                         int floor_height, int ceiling_height, vertex_t *start_vertex, vertex_t *end_vertex,
                         int line_flags) {
    if (texture_number == 0) {
        return nullptr;
    }

    auto start_x = RT_FixedToFloating(start_vertex->x);
    auto start_y = RT_FixedToFloating(start_vertex->y);

    auto end_x = RT_FixedToFloating(end_vertex->x);
    auto end_y = RT_FixedToFloating(end_vertex->y);

    auto bottom_left = glm::vec3(start_x, RT_FixedToFloating(floor_height), start_y);
    auto top_left = glm::vec3(start_x, RT_FixedToFloating(ceiling_height), start_y);

    auto bottom_right = glm::vec3(end_x, RT_FixedToFloating(floor_height), end_y);
    auto top_right = glm::vec3(end_x, RT_FixedToFloating(ceiling_height), end_y);

    auto texture = get_device_texture(texture_number, wad, graphics_data);

    auto horizontal_len = glm::length(top_right - top_left);
    auto vertical_len = glm::length(bottom_left - top_left);
    auto uv_scale = glm::vec2(static_cast<float>(glm::length(top_right - top_left) / texture->width()),
                              static_cast<float>(glm::length(bottom_left - top_left) / texture->height()));
    uv_scale /= glm::vec2(horizontal_len, vertical_len);

    auto square = create_device_type<Square>(top_left, top_right - top_left, bottom_left - top_left, uv_scale, texture);

    if (line_flags & ML_DONTPEGBOTTOM) {
        square->uv_offset = texture->height() - vertical_len;
    }

    return square;
}

Square *create_sector_adjacent_wall(short texture_number,
                                    wad::Wad &wad, wad::GraphicsData &graphics_data,
                                    int front_sector_height, int back_sector_height,
                                    vertex_t *start_vertex,
                                    vertex_t *end_vertex) {
    if (texture_number == 0) {
        return nullptr;
    }

    auto lower_height = RT_FixedToFloating(front_sector_height);
    auto higher_height = RT_FixedToFloating(back_sector_height);
    if (lower_height > higher_height) {
        std::swap(lower_height, higher_height);
    }

    auto start_x = RT_FixedToFloating(start_vertex->x);
    auto start_y = RT_FixedToFloating(start_vertex->y);

    auto end_x = RT_FixedToFloating(end_vertex->x);
    auto end_y = RT_FixedToFloating(end_vertex->y);

    auto bottom_left = glm::vec3(start_x, lower_height, start_y);
    auto top_left = glm::vec3(start_x, higher_height, start_y);

    auto bottom_right = glm::vec3(end_x, lower_height, end_y);
    auto top_right = glm::vec3(end_x, higher_height, end_y);

    auto texture = get_device_texture(texture_number, wad, graphics_data);

    auto horizontal_len = glm::length(top_right - top_left);
    auto vertical_len = glm::length(bottom_left - top_left);
    auto uv_scale = glm::vec2(static_cast<float>(glm::length(top_right - top_left) / texture->width()),
                              static_cast<float>(glm::length(bottom_left - top_left) / texture->height()));
    uv_scale /= glm::vec2(horizontal_len, vertical_len);


    return create_device_type<Square>(top_left, top_right - top_left, bottom_left - top_left, uv_scale,
                                      texture);
}

// Given three collinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
bool on_segment(glm::vec2 p, glm::vec2 q, glm::vec2 r) {
    if (q.x <= glm::max(p.x, r.x) && q.x >= glm::min(p.x, r.x) &&
        q.y <= glm::max(p.y, r.y) && q.y >= glm::min(p.y, r.y))
        return true;

    return false;
}

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(glm::vec2 p, glm::vec2 q, glm::vec2 r) {
    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // collinear

    return (val > 0) ? 1 : 2; // clock or counterclock wise
}

// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
bool lines_intersect(glm::vec2 p1, glm::vec2 q1, glm::vec2 p2, glm::vec2 q2) {
    // Find the four orientations needed for general and
    // special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // General case
    if (o1 != o2 && o3 != o4)
        return true;

    // Special Cases
    // p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0 && on_segment(p1, p2, q1)) return true;

    // p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0 && on_segment(p1, q2, q1)) return true;

    // p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0 && on_segment(p2, p1, q2)) return true;

    // p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0 && on_segment(p2, q1, q2)) return true;

    return false; // Doesn't fall in any of the above cases
}

bool is_point_inside_polygon(const Polygon &container, const glm::vec2 &point) {
    // Test all edges of the (possibly) containing polygon against an arbitrary line starting from the given point
    // NOTE: I am doing the check twice, with two different "abritrary lines". This is to avoid false positives when the lines become collinear with lines in the polygon.
    auto end_point1 = point + glm::vec2{0, -2000};
    auto end_point2 = point + glm::vec2{0, 2000};

    int intersections1 = 0;
    int intersections2 = 0;
    for (int i = 0; i < container.size(); ++i) {
        auto j = (i == container.size() - 1) ? 0 : i + 1;

        auto edge_start = container[i];
        auto edge_end = container[j];

        if (lines_intersect(point, end_point1, edge_start, edge_end)) {
            intersections1++;
        }

        if (lines_intersect(point, end_point2, edge_start, edge_end)) {
            intersections2++;
        }
    }

    return (intersections1 & 1) > 0 && (intersections2 & 1) > 0;
}

bool is_polygon_inside_other(const Polygon &container, const Polygon &test_polygon) {

    for (auto &p: test_polygon) {
        if (is_point_inside_polygon(container, p)) {
            return true;
        }
    }

    return false;
}

void combine_polygons(std::vector<Polygon> &polygons, size_t parent_polygon_index, size_t child_polygon_index, std::vector<bool>& parent_polygon_weld_points) {
    auto &parent_polygon = polygons[parent_polygon_index];
    auto &child_polygon = polygons[child_polygon_index];

    auto is_parent_cw = is_polygon_cw_winding(parent_polygon);

    auto is_child_cw = is_polygon_cw_winding(child_polygon);

    if (!is_parent_cw) {
        std::reverse(parent_polygon.begin(), parent_polygon.end());
    }

    if (is_child_cw) {
        std::reverse(child_polygon.begin(), child_polygon.end());
    }

    auto found_vertex_pair = false;
    int parent_vertex = -1;
    int child_vertex = -1;
    float best_distance = FLT_MAX;
    for (int i = 0; i < child_polygon.size(); ++i) {
        for (auto j = 0; j < parent_polygon.size(); ++j) {
            auto distance = glm::length(child_polygon[i] - parent_polygon[j]);
            auto is_already_weld_point = parent_polygon_weld_points[j];
            if(is_already_weld_point) {
                distance *= 1000; // Dont diregard the point entirely, but make it less likely to be choosen.
            }
            if (distance < best_distance) {
                best_distance = distance;
                parent_vertex = j;
                child_vertex = i;
            }
        }
    }

    if (parent_vertex == -1) {
        printf("ABORTING. Cant find best parent/child vertex for polygon combination\n");
        exit(-1);
    }
    parent_polygon_weld_points[parent_vertex] = true;
    auto cv = child_polygon[child_vertex];
    std::rotate(child_polygon.begin(), child_polygon.begin() + child_vertex, child_polygon.end());
    child_polygon.push_back(cv);
    child_polygon.push_back(parent_polygon[parent_vertex]);
    //auto first_in_child = child_polygon[child_vertex];

    //child_polygon.insert(child_polygon.begin() + child_vertex, first_in_child);
    //child_polygon.insert(child_polygon.begin() + child_vertex, parent_polygon[parent_vertex]);
    //child_polygon.push_back(first_in_child);
    //

    parent_polygon.insert(parent_polygon.begin() + parent_vertex + 1, child_polygon.begin(),
                          child_polygon.end());

    std::vector<bool> new_points(child_polygon.size(), false);
    parent_polygon_weld_points.insert(parent_polygon_weld_points.begin() + parent_vertex + 1, new_points.begin(), new_points.end());
    parent_polygon_weld_points[parent_vertex + 1] = true;
    parent_polygon_weld_points[parent_vertex + child_polygon.size()] = true;
    parent_polygon_weld_points[parent_vertex + child_polygon.size() - 1] = true;
}

bool is_polygon_cw_winding(const std::vector<glm::vec2> &polygon) {

    float total_sum = 0.0f;
    for (int i = 0; i < polygon.size(); ++i) {
        auto next_index = i + 1;
        if (i == polygon.size() - 1) {
            next_index = 0;
        }

        total_sum += (polygon[next_index].x - polygon[i].x) * (polygon[next_index].y + polygon[i].y);
    }

    return total_sum >= 0.0f;
}

void create_mesh_from_polygon(
        sector_t *sector,
        SceneData &scene_data,
        std::vector<glm::vec2> &polygon) {
    if (!is_polygon_cw_winding(polygon)) {
        std::reverse(polygon.begin(), polygon.end());
    }

    auto floor_texture = get_device_texture_from_flat(sector->floorpic);

    std::vector<glm::vec3> polys3d;
    for (auto &p: polygon) {
        polys3d.emplace_back(p.x, RT_FixedToFloating(sector->floorheight), p.y);
    }

    auto triangles = triangulate_polygon(polys3d, floor_texture);

    auto &sector_geometry = scene_data.sector_geometry[sector];
    scene_data.sector_geometry[sector].floor.insert(sector_geometry.floor.end(), triangles.begin(), triangles.end());
    scene_data.triangles.insert(scene_data.triangles.end(), triangles.begin(), triangles.end());

    if (sector->ceilingpic != skyflatnum) {//sector.ceiling_texture != "F_SKY1") {
        auto ceiling_texture = get_device_texture_from_flat(sector->ceilingpic);

        std::vector<Triangle *> ceiling_triangles;
        for (auto &triangle: triangles) {
            auto ceiling_triangle = create_device_type<Triangle>(triangle->v0, triangle->v1, triangle->v2,
                                                                 ceiling_texture);
            ceiling_triangle->v0.y = ceiling_triangle->v1.y = ceiling_triangle->v2.y = RT_FixedToFloating(
                    sector->ceilingheight);
            ceiling_triangles.push_back(ceiling_triangle);
        }

        sector_geometry.ceiling.insert(sector_geometry.ceiling.end(), ceiling_triangles.begin(),
                                       ceiling_triangles.end());
        scene_data.triangles.insert(scene_data.triangles.end(), ceiling_triangles.begin(), ceiling_triangles.end());
    }
}

DeviceTexture *get_device_texture(short texture_number,
                                  wad::Wad &wad,
                                  wad::GraphicsData &graphics_data) {
    auto texture = detail::texture_cache.get_texture(texture_number);
    if (texture) {
        return texture;
    }
    size_t texture_width{};
    size_t texture_height{};


    const auto &tex = graphics_data.get_texture(texture_number);
    auto pixels = tex.get_pixels(wad, graphics_data.patch_names());
    texture = create_device_type<DeviceTexture>(pixels, tex.width(), tex.height());

    detail::texture_cache.insert_texture(texture_number, texture);

    return texture;
}

DeviceTexture *get_device_texture_from_flat(short flatnum) {
    auto texture = detail::texture_cache.get_texture_for_flat(flatnum);
    if (texture) {
        return texture;
    }

    texture = create_device_texture_from_flat(firstflat + flatnum);
    detail::texture_cache.insert_flat_texture(flatnum, texture);

    return texture;
}

DeviceTexture *create_device_texture_from_flat(intptr_t lump_number) {
    auto lump_size = W_LumpLength(lump_number);
    auto lump_data = reinterpret_cast<unsigned char *>(W_CacheLumpNum(lump_number, PU_CACHE));

    std::vector<std::uint16_t> pixels(lump_size, 0);
    for (int i = 0; i < lump_size; ++i) {
        pixels[i] = lump_data[i];
    }

    // std::cout << "creating texture " << lump.name << std::endl;
    return create_device_type<DeviceTexture>(pixels, 64, 64);
}

DeviceTexture *create_device_texture_from_map_texture(wad::Wad &wad, const wad::MapTexture &texture,
                                                      const std::vector<std::string> &patch_names) {
    auto pixels = texture.get_pixels(wad, patch_names);

    std::cout << "creating texture " << texture.name() << std::endl;
    return create_device_type<DeviceTexture>(pixels, texture.width(),
                                             texture.height());
}

void RT_ChangeSideTopTexture(side_t *side, int texture_num) {
    detail::sidedef_to_wall_lookup[side].top->texture = get_device_texture(texture_num, *detail::wad,
                                                                           *detail::graphics_data);
}

void RT_ChangeSideMidTexture(side_t *side, int texture_num) {
    detail::sidedef_to_wall_lookup[side].middle->texture = get_device_texture(texture_num, *detail::wad,
                                                                              *detail::graphics_data);
}

void RT_ChangeSideBottomTexture(side_t *side, int texture_num) {
    detail::sidedef_to_wall_lookup[side].bottom->texture = get_device_texture(texture_num, *detail::wad,
                                                                              *detail::graphics_data);
}

void debug_polygon(Polygon &polygon, const std::string& name) {
    constexpr size_t image_width = 1024;
    constexpr size_t image_height = 1024;
    auto image = FreeImage_Allocate(1024, 1024, 32);

    for (int y = 0; y < image_height; ++y) {
        for (int x = 0; x < image_width; ++x) {
            RGBQUAD rgb;
            rgb.rgbBlue = rgb.rgbGreen = rgb.rgbRed = rgb.rgbReserved = 255;

            FreeImage_SetPixelColor(image, x, y, &rgb);
        }
    }

    auto smallest = glm::vec2(FLT_MAX, FLT_MAX);
    auto largest = glm::vec2(FLT_MIN, FLT_MIN);

    for (auto vertex: polygon) {
        smallest = glm::min(smallest, vertex);
        largest = glm::max(largest, vertex);
    }

    smallest = smallest - glm::vec2(10.0, 10.0);
    largest = largest + glm::vec2(10.0, 10.0);

    auto polygon_size = largest - smallest;

    auto get_slope = [](glm::vec2 &start, glm::vec2 &end) -> std::optional<float> {
        if (start.x == end.x) {
            return std::nullopt;
        }

        auto slope = (end.y - start.y) / (end.x - start.x);
        if (glm::abs(slope) > 100000.0f) {
            // slope is steep enough to handle as vertical
            return std::nullopt;
        }

        return slope;
    };

    auto get_intercept = [](glm::vec2 &start, std::optional<float> &slope) -> float {
        if (slope.has_value()) {
            return start.y - slope.value() * start.x;
        }
        return start.x;
    };

    for (int i = 0; i < polygon.size(); ++i) {
        auto next_index = (i + 1) % polygon.size();
        auto from_vertex = polygon[i];
        auto to_vertex = polygon[next_index];

        auto start = glm::vec2(
                ((from_vertex.x - smallest.x) / polygon_size.x) * (image_width - 1.0),
                ((from_vertex.y - smallest.y) / polygon_size.y) * (image_height - 1.0)
        );

        auto end = glm::vec2(
                ((to_vertex.x - smallest.x) / polygon_size.x) * (image_width - 1.0),
                ((to_vertex.y - smallest.y) / polygon_size.y) * (image_height - 1.0)
        );

        auto slope = get_slope(start, end);
        auto intercept = get_intercept(start, slope);

        auto previous_distance = FLT_MAX;

        auto current_pos = glm::vec2(start.x, start.y);

        while(glm::length(current_pos - end) < previous_distance) {
            previous_distance = glm::length(current_pos - end);

            auto base_increment = 0.1f;
            if(slope.has_value()) {
                auto step_increment = start.x > end.x ? -base_increment : base_increment;
                auto diff = end.x - start.x;
                if(glm::abs(diff) < glm::abs(step_increment)) {
                    step_increment = diff;
                }
                current_pos.x += step_increment;
                current_pos.y = slope.value() * current_pos.x + intercept;
            }
            else {
                auto step_increment = start.y > end.y ?  -base_increment :  base_increment ;
                auto diff = end.y - start.y;
                if (glm::abs(diff) < glm::abs(step_increment)) {
                    step_increment = diff;
                }
                current_pos.y += step_increment;
            }

            RGBQUAD rgb;
            rgb.rgbRed = rgb.rgbGreen = rgb.rgbBlue = 0;
            rgb.rgbReserved = 255;
            FreeImage_SetPixelColor(image, (int)current_pos.x, (int)current_pos.y, &rgb);
        }
    }

    FreeImage_Save(FIF_PNG, image, fmt::format("/home/emil/doom_wads/sectors/polygon_{}.png", name).c_str());
}