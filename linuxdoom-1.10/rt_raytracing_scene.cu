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
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
#include <set>

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

struct MovableSector {
    std::vector<Square *> ceiling_walls;
    std::vector<Square *> floor_walls;
    std::vector<Triangle *> ceiling;
    std::vector<Triangle *> floor;
};


struct SceneData {
    std::vector<Square *> walls;
    std::vector<Triangle *> triangles;
    std::vector<SceneEntity *> entities;
    std::vector<DeviceTexture *> device_textures;
    std::unordered_map<sector_t *, MovableSector> movable_sector;
};

typedef std::vector<glm::vec2> Polygon;

bool on_segment(glm::vec2 p, glm::vec2 q, glm::vec2 r);

int orientation(glm::vec2 p, glm::vec2 q, glm::vec2 r);

bool lines_intersect(glm::vec2 p1, glm::vec2 q1, glm::vec2 p2, glm::vec2 q2);

bool is_point_inside_polygon(const Polygon &container, const glm::vec2 &point);

bool is_polygon_inside_other(const Polygon &container, const Polygon &test_polygon);

void combine_polygons(std::vector<Polygon> &polygons, size_t parent_polygon_index, size_t child_polygon_index);

bool is_polygon_cw_winding(const std::vector<glm::vec2> &polygon);

void create_mesh_from_polygon(
        sector_t *sector,
        TextureCache &texture_cache,
        SceneData &scene_data,
        std::vector<glm::vec2> &polygon,
        bool is_moving_sector,
        int sector_number);

DeviceTexture *get_device_texture(short texture_number,
                                  wad::Wad &wad,
                                  wad::GraphicsData &graphics_data,
                                  TextureCache &texture_cache);

DeviceTexture *get_device_texture_from_flat(short flatnum, TextureCache &texture_cache);

DeviceTexture *create_device_texture_from_flat(intptr_t lump_number);

Square *create_sector_adjacent_wall(short texture_number,
                                    wad::Wad &wad, wad::GraphicsData &graphics_data,
                                    TextureCache &texture_cache,
                                    int front_sector_height, int back_sector_height,
                                    vertex_t *start_vertex,
                                    vertex_t *end_vertex);

Square *create_main_wall(short texture_number, wad::Wad &wad, wad::GraphicsData &graphics_data,
                         TextureCache &texture_cache,
                         int floor_height, int ceiling_height, vertex_t *start_vertex, vertex_t *end_vertex);

Scene *RT_BuildScene(wad::Wad &wad, wad::GraphicsData &graphics_data) {

    TextureCache texture_cache;
    SceneData scene_data;

    std::set<sector_t *> moving_sectors;
    line_t *line_ptr = lines;
    for (int i = 0; i < numlines; i++, line_ptr++) {
        if (line_ptr->special == 117) {
            if (line_ptr->sidenum[1] == -1) { // Assuming sidenum[1] is left?
                continue;
            }

            int asd = line_ptr->sidenum[1];
            auto &side = sides[line_ptr->sidenum[1]];
            moving_sectors.insert(side.sector);
        }
    }

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
                        combine_polygons(polygons, i, it->second);
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

            create_mesh_from_polygon(
                    sector_ptr,
                    texture_cache,
                    scene_data,
                    polygons[i],
                    moving_sectors.find(sector_ptr) != moving_sectors.end(),
                    sector_number);
        }

    }

    // int i = -1;
    line_ptr = lines;
    for (int i = 0; i < numlines; i++, line_ptr++) {
        //++i;

        auto start_vertex = line_ptr->v1;
        auto end_vertex = line_ptr->v2;

        // REminder: line_ptr->sidenum[1] is left side
        if (line_ptr->sidenum[0] > -1 && line_ptr->sidenum[1] > -1) {
            auto left_side = &sides[line_ptr->sidenum[1]];
            auto right_side = &sides[line_ptr->sidenum[0]];

            auto left_sector = left_side->sector;
            auto right_sector = right_side->sector;

            auto left_upper_wall = create_sector_adjacent_wall(
                    left_side->toptexture,
                    wad,
                    graphics_data,
                    texture_cache,
                    left_sector->ceilingheight,
                    right_sector->ceilingheight,
                    start_vertex,
                    end_vertex);

            if (left_upper_wall) {
                // if (line.special_type == 117) { // Vertical door
                if (moving_sectors.find(left_side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[left_side->sector].ceiling_walls.push_back(left_upper_wall);
                }

                if (moving_sectors.find(right_side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[right_side->sector].ceiling_walls.push_back(left_upper_wall);
                }


                scene_data.walls.push_back(left_upper_wall);
            }

            auto right_upper_wall = create_sector_adjacent_wall(
                    right_side->toptexture,
                    wad,
                    graphics_data,
                    texture_cache,
                    left_sector->ceilingheight,
                    right_sector->ceilingheight,
                    start_vertex,
                    end_vertex);

            if (right_upper_wall) {
                //if (line.special_type == 117) { // Vertical door
                if (moving_sectors.find(left_side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[left_side->sector].ceiling_walls.push_back(right_upper_wall);
                }

                if (moving_sectors.find(right_side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[right_side->sector].ceiling_walls.push_back(right_upper_wall);
                }

                scene_data.walls.push_back(right_upper_wall);
            }

            auto left_lower_wall = create_sector_adjacent_wall(
                    left_side->bottomtexture,
                    wad,
                    graphics_data,
                    texture_cache,
                    left_sector->floorheight,
                    right_sector->floorheight,
                    start_vertex,
                    end_vertex);

            if (left_lower_wall) {
                //if (line.special_type == 117) { // Vertical door
                if (moving_sectors.find(left_side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[left_side->sector].floor_walls.push_back(left_lower_wall);
                }

                if (moving_sectors.find(right_side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[right_side->sector].floor_walls.push_back(left_lower_wall);
                }
                scene_data.walls.push_back(left_lower_wall);
            }
            auto right_lower_wall = create_sector_adjacent_wall(
                    right_side->bottomtexture,
                    wad,
                    graphics_data,
                    texture_cache,
                    left_sector->floorheight,
                    right_sector->floorheight,
                    start_vertex,
                    end_vertex);

            if (right_lower_wall) {
                //if (line.special_type == 117) { // Vertical door
                if (moving_sectors.find(left_side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[left_side->sector].floor_walls.push_back(right_lower_wall);
                }

                if (moving_sectors.find(right_side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[right_side->sector].floor_walls.push_back(right_lower_wall);
                }
                scene_data.walls.push_back(right_lower_wall);
            }

        }

        // left side
        if (line_ptr->sidenum[1] > -1) {
            auto side = &sides[line_ptr->sidenum[1]];
            auto sector = side->sector;

            auto wall = create_main_wall(
                    side->midtexture,
                    wad,
                    graphics_data,
                    texture_cache,
                    sector->floorheight,
                    sector->ceilingheight,
                    start_vertex,
                    end_vertex);

            if (wall) {
                //if (line.special_type == 117) { // Vertical door
                if (moving_sectors.find(side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[side->sector].floor_walls.push_back(wall);
                }
                scene_data.walls.push_back(wall);
            }

            if (wall) {
                scene_data.walls.push_back(wall);
            }
        }

        if (line_ptr->sidenum[0] > -1) {
            auto side = &sides[line_ptr->sidenum[0]];
            auto sector = side->sector;

            auto wall = create_main_wall(
                    side->midtexture,
                    wad,
                    graphics_data,
                    texture_cache,
                    sector->floorheight,
                    sector->ceilingheight,
                    start_vertex,
                    end_vertex);

            if (wall) {
                //if (line.special_type == 117) { // Vertical door
                if (moving_sectors.find(side->sector) != moving_sectors.end()) {
                    scene_data.movable_sector[side->sector].floor_walls.push_back(wall);
                }
                scene_data.walls.push_back(wall);
            }

            if (wall) {
                scene_data.walls.push_back(wall);
            }
        }
    }

    auto sky_texture = get_device_texture(skytexture, wad, graphics_data, texture_cache);

    return create_device_type<Scene>(scene_data.walls, scene_data.triangles, scene_data.entities, sky_texture);
}

Square *create_main_wall(short texture_number, wad::Wad &wad, wad::GraphicsData &graphics_data,
                         TextureCache &texture_cache,
                         int floor_height, int ceiling_height, vertex_t *start_vertex, vertex_t *end_vertex) {
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

    auto texture = get_device_texture(texture_number, wad, graphics_data, texture_cache);

    auto horizontal_len = glm::length(top_right - top_left);
    auto vertical_len = glm::length(bottom_left - top_left);
    auto uv_scale = glm::vec2(static_cast<float>(glm::length(top_right - top_left) / texture->width()),
                              static_cast<float>(glm::length(bottom_left - top_left) / texture->height()));
    uv_scale /= glm::vec2(horizontal_len, vertical_len);

    return create_device_type<Square>(top_left, top_right - top_left, bottom_left - top_left, uv_scale, texture);
}

Square *create_sector_adjacent_wall(short texture_number,
                                    wad::Wad &wad, wad::GraphicsData &graphics_data,
                                    TextureCache &texture_cache,
                                    int front_sector_height, int back_sector_height,
                                    vertex_t *start_vertex,
                                    vertex_t *end_vertex) {
    if (texture_number == 0) {
        return nullptr;
    }
    if (front_sector_height == back_sector_height) {
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

    auto texture = get_device_texture(texture_number, wad, graphics_data, texture_cache);

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

void combine_polygons(std::vector<Polygon> &polygons, size_t parent_polygon_index, size_t child_polygon_index) {
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
        TextureCache &texture_cache,
        SceneData &scene_data,
        std::vector<glm::vec2> &polygon,
        bool is_moving_sector,
        int sector_number) {

    auto foo = is_triangle_cw_winding(polygon[0], polygon[1], polygon[2]);
    auto bar = is_polygon_cw_winding(polygon);

    if (!is_polygon_cw_winding(polygon)) {
        std::reverse(polygon.begin(), polygon.end());
    }

    auto floor_texture = get_device_texture_from_flat(sector->floorpic, texture_cache);
    /*auto floor_texture = get_device_texture(sector.floor_texture, wad, graphics_data,
                                            scene_data.device_textures, device_texture_lookup, true);*/


    std::vector<glm::vec3> polys3d;
    // polys3d.reserve(polygon.size());
    for (auto &p: polygon) {
        polys3d.emplace_back(p.x, RT_FixedToFloating(sector->floorheight), p.y);
    }
    auto triangles = triangulate_polygon(polys3d, floor_texture);

    if (is_moving_sector) {
        auto &movable_sector = scene_data.movable_sector[sector];
        movable_sector.floor.insert(movable_sector.floor.end(), triangles.begin(), triangles.end());
    }

    scene_data.triangles.insert(scene_data.triangles.end(), triangles.begin(), triangles.end());

    if (sector->ceilingpic != skyflatnum) {//sector.ceiling_texture != "F_SKY1") {
        auto ceiling_texture = get_device_texture_from_flat(sector->ceilingpic, texture_cache);
        /*auto ceiling_texture = get_device_texture(sector.ceiling_texture, wad, graphics_data,
                                                  scene_data.device_textures, device_texture_lookup,
                                                  true);*/


        std::vector<Triangle *> ceiling_triangles;
        for (auto &triangle: triangles) {
            auto ceiling_triangle = create_device_type<Triangle>(triangle->v0, triangle->v1, triangle->v2,
                                                                 ceiling_texture);
            ceiling_triangle->v0.y = ceiling_triangle->v1.y = ceiling_triangle->v2.y = RT_FixedToFloating(
                    sector->ceilingheight);
            ceiling_triangles.push_back(ceiling_triangle);
        }

        if (is_moving_sector) {
            auto &movable_sector = scene_data.movable_sector[sector];
            movable_sector.ceiling.insert(movable_sector.ceiling.end(), triangles.begin(), triangles.end());
        }

        scene_data.triangles.insert(scene_data.triangles.end(), ceiling_triangles.begin(), ceiling_triangles.end());
    }
}

DeviceTexture *get_device_texture(short texture_number,
                                  wad::Wad &wad,
                                  wad::GraphicsData &graphics_data,
                                  TextureCache &texture_cache) {
    auto texture = texture_cache.get_texture(texture_number);
    if (texture) {
        return texture;
    }
    size_t texture_width{};
    size_t texture_height{};


    const auto &tex = graphics_data.get_texture(texture_number);
    auto pixels = tex.get_pixels(wad, graphics_data.patch_names());
    texture = create_device_type<DeviceTexture>(pixels, tex.width(), tex.height());

    texture_cache.insert_texture(texture_number, texture);

    return texture;
}

DeviceTexture *get_device_texture_from_flat(short flatnum, TextureCache &texture_cache) {
    auto texture = texture_cache.get_texture_for_flat(flatnum);
    if (texture) {
        return texture;
    }

    texture = create_device_texture_from_flat(firstflat + flatnum);
    texture_cache.insert_flat_texture(flatnum, texture);

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
