#include "rt_raytracing.cuh"
#include "r_state.h"
#include "z_zone.h" // FOR PU_CACHE
#include "w_wad.h"
#include "renderer/device_texture.cuh"
#include "renderer/cuda_utils.cuh"
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>

typedef std::vector<glm::vec2> Polygon;

bool on_segment(glm::vec2 p, glm::vec2 q, glm::vec2 r);
int orientation(glm::vec2 p, glm::vec2 q, glm::vec2 r);
bool lines_intersect(glm::vec2 p1, glm::vec2 q1, glm::vec2 p2, glm::vec2 q2);
bool is_point_inside_polygon(const Polygon &container, const glm::vec2 &point);
bool is_polygon_inside_other(const Polygon &container, const Polygon &test_polygon);
void combine_polygons(std::vector<Polygon> &polygons, size_t parent_polygon_index, size_t child_polygon_index);
bool is_polygon_cw_winding(const std::vector<glm::vec2> &polygon);

DeviceTexture *create_device_texture_from_flat(intptr_t lump_number);

void RT_BuildScene() {
    sector_t *sector_ptr = sectors;


    for(int i = 0; i < numsectors; i++, sector_ptr++) {

        side_t *side_ptr = sides;
        std::vector<int> all_sides;
        for(int j = 0; j < numsides; j++, side_ptr++) {
            if(side_ptr->sector == sector_ptr) {
                all_sides.push_back(j);
            }
        }

        line_t *line_ptr = lines;
        std::vector<line_t> all_lines;
        for(int j = 0; j < numlines; j++, line_ptr++) {
            for(auto side: all_sides) {
                if(line_ptr->sidenum[0] == side || line_ptr->sidenum[1] == side) { // 0 is right?
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
                polygon.emplace_back(glm::vec2{line.v1->x, line.v1->y});
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

            /*create_mesh_from_polygon(
                    sector0,
                    wad,
                    graphics,
                    device_texture_lookup,
                    data,
                    polygons[i]
            );*/
        }

    }
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
/*
void create_mesh_from_polygon(
        wad::Sector &sector,
        wad::Wad &wad,
        wad::GraphicsData &graphics_data,
        std::unordered_map<std::string, size_t> &device_texture_lookup,
        WadData &data,
        std::vector<glm::vec2> &polygon) {

    if (!is_polygon_cw_winding(polygon)) {
        std::reverse(polygon.begin(), polygon.end());
    }

    auto floor_texture = get_device_texture(sector.floor_texture, wad, graphics_data,
                                            data.device_textures, device_texture_lookup, true);


    std::vector<glm::vec3> polys3d;
    // polys3d.reserve(polygon.size());
    for (auto &p: polygon) {
        polys3d.emplace_back(p.x, sector.floor_height, p.y);
    }
    auto triangles = triangulate_polygon(polys3d, floor_texture);

    data.triangles.insert(data.triangles.end(), triangles.begin(), triangles.end());

    if (sector.ceiling_texture != "-" && sector.ceiling_texture != "F_SKY1") {
        auto ceiling_texture = get_device_texture(sector.ceiling_texture, wad, graphics_data,
                                                  data.device_textures, device_texture_lookup,
                                                  true);
        for (auto &triangle: triangles) {
            triangle.v0.y = triangle.v1.y = triangle.v2.y = sector.ceiling_height;
            triangle.texture = ceiling_texture;
        }

        data.triangles.insert(data.triangles.end(), triangles.begin(), triangles.end());
    }
}
*/

/*
DeviceTexture *get_device_texture(const std::string &texture_name,
                                  wad::Wad &wad,
                                  wad::GraphicsData &graphics_data,
                                  std::vector<DeviceTexture *> &device_textures,
                                  std::unordered_map<std::string, size_t> &texture_lookup,
                                  bool is_flat) {
    auto it = texture_lookup.find(texture_name);
    if (it != texture_lookup.end()) {
        return device_textures[it->second];
    }

    DeviceTexture *device_texture;

    size_t texture_width{};
    size_t texture_height{};
    if (is_flat) {
        auto lump = W_GetNumForName(texture_name.c_str());
        device_texture = create_device_texture_from_flat(lump);
    } else {
        if (texture_name == "A-VINE3") {
            int foo = 2;
        }
        auto texture_num = R_TextureNumForName(texture_name.c_str());
        const auto &tex = graphics_data.get_texture(texture_name);
        // R_CheckTextureNumForName(texture_name.c_str());
        device_texture = create_device_texture_from_map_texture(wad, tex, graphics_data.patch_names());
    }


    auto new_index = device_textures.size();
    device_textures.push_back(device_texture);
    texture_lookup.insert({texture_name, new_index});

    return device_texture;
}

DeviceTexture *create_device_texture_from_flat(intptr_t lump_number) {
    auto lump_size = W_LumpLength(lump_number);
    auto lump_data = reinterpret_cast<unsigned char*>(W_CacheLumpNum(lump_number, PU_CACHE));

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
 */