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
#include "geometry/polygon.h"
#include "geometry/triangulation.h"
#include "rt_material.cuh"
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

void create_mesh_from_polygon(
        sector_t *sector,
        SceneData &scene_data,
        geometry::Polygon &polygon);

DeviceMaterial get_device_material(short texture_number,
                                   wad::Wad &wad,
                                   wad::GraphicsData &graphics_data);

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
        std::vector<geometry::Polygon> polygons;
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

            std::vector<glm::vec2> vertices;
            for (auto &line: sorted_lines) {
                vertices.emplace_back(glm::vec2{RT_FixedToFloating(line.v1->x), RT_FixedToFloating(line.v1->y)});
            }
            polygons.emplace_back(vertices);
        }

        // BUILD A MAP OF PARENT-CHILD RELATIONSHIPS
        std::unordered_multimap<size_t, size_t> parent_map;
        for (int i = 0; i < polygons.size(); ++i) {
            auto &current_polygon = polygons[i];

            for (int j = 0; j < polygons.size(); ++j) {
                if (i == j) {
                    continue;
                }

                // if (is_polygon_inside_other(current_polygon, polygons[j])) {
                if (current_polygon.intersects_polygon(polygons[j])) {
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
            for (int i = 0; i < polygons.size(); ++i) {
                if (polygon_is_pruned[i]) {
                    continue;
                }

                auto children = parent_map.equal_range(i);
                for (auto it = children.first; it != children.second; ++it) {
                    if (is_direct_ancestor(i, it->second, parent_map) && is_childless(it->second, parent_map)) {

                        polygons[i].assert_winding(geometry::Winding::Clockwise);
                        polygons[it->second].assert_winding(geometry::Winding::CounterClockwise);
                        polygons[i].combine_with(polygons[it->second]);

                        if (false) {
                            polygons[i].debug_image(fmt::format("combine_{}_{}_{}", sector_number, i, it->second));
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

        if (i == 220) {
            int ppp = 43;
        }

        // REminder: line_ptr->sidenum[1] is left side
        if (line_ptr->sidenum[0] > -1 && line_ptr->sidenum[1] > -1) {
            auto left_side = &sides[line_ptr->sidenum[1]];
            auto right_side = &sides[line_ptr->sidenum[0]];

            auto left_sector = left_side->sector;
            auto right_sector = right_side->sector;

            auto left_tag = left_sector->tag;
            auto right_tag = right_sector->tag;
            if (left_tag == 14 || right_tag == 14) {
                int dffdfg = 43;
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
            if (sector->tag == 14) {
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

            if (sector->tag == 14) {
                int fgfgf = 43;
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

    auto material = get_device_material(texture_number, wad, graphics_data);

    auto horizontal_len = glm::length(top_right - top_left);
    auto vertical_len = glm::length(bottom_left - top_left);
    auto uv_scale = glm::vec2(static_cast<float>(glm::length(top_right - top_left) / material.diffuse_map()->width()),
                              static_cast<float>(glm::length(bottom_left - top_left) / material.diffuse_map()->height()));
    uv_scale /= glm::vec2(horizontal_len, vertical_len);

    auto square = create_device_type<Square>(top_left, top_right - top_left, bottom_left - top_left, uv_scale, material);

    if (line_flags & ML_DONTPEGBOTTOM) {
        square->uv_offset = material.diffuse_map()->height() - vertical_len;
        square->texture_wrapping = false;
        square->lower_unpegged = true;
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

    auto material = get_device_material(texture_number, wad, graphics_data);

    auto horizontal_len = glm::length(top_right - top_left);
    auto vertical_len = glm::length(bottom_left - top_left);
    auto uv_scale = glm::vec2(static_cast<float>(glm::length(top_right - top_left) / material.diffuse_map()->width()),
                              static_cast<float>(glm::length(bottom_left - top_left) / material.diffuse_map()->height()));
    uv_scale /= glm::vec2(horizontal_len, vertical_len);


    return create_device_type<Square>(top_left, top_right - top_left, bottom_left - top_left, uv_scale,
                                      material);
}

void create_mesh_from_polygon(
        sector_t *sector,
        SceneData &scene_data,
        geometry::Polygon &polygon) {
    polygon.assert_winding(geometry::Winding::CounterClockwise);

    auto floor_texture = get_device_texture_from_flat(sector->floorpic);

    std::vector<glm::vec3> polys3d;
    for (auto &p: polygon.vertices()) {
        polys3d.emplace_back(p.x, RT_FixedToFloating(sector->floorheight), p.y);
    }

    auto triangles_2d = geometry::triangulate(polygon);

    auto floor_height = RT_FixedToFloating(sector->floorheight);
    std::vector<Triangle *> floor_triangles;
    floor_triangles.reserve(triangles_2d.size());
    for (auto &tri: triangles_2d) {
        floor_triangles.push_back(create_device_type<Triangle>(
                glm::vec3(tri.v0().x, floor_height, tri.v0().y),
                glm::vec3(tri.v1().x, floor_height, tri.v1().y),
                glm::vec3(tri.v2().x, floor_height, tri.v2().y),
                DeviceMaterial(floor_texture)));
    }

    auto &sector_geometry = scene_data.sector_geometry[sector];
    scene_data.sector_geometry[sector].floor.insert(sector_geometry.floor.end(), floor_triangles.begin(),
                                                    floor_triangles.end());
    scene_data.triangles.insert(scene_data.triangles.end(), floor_triangles.begin(), floor_triangles.end());

    if (sector->ceilingpic != skyflatnum) {//sector.ceiling_texture != "F_SKY1") {
        auto ceiling_texture = get_device_texture_from_flat(sector->ceilingpic);

        auto ceiling_height = RT_FixedToFloating(sector->ceilingheight);
        std::vector<Triangle *> ceiling_triangles;
        ceiling_triangles.reserve(triangles_2d.size());
        for (auto &tri: triangles_2d) {
            ceiling_triangles.push_back(create_device_type<Triangle>(
                    glm::vec3(tri.v0().x, ceiling_height, tri.v0().y),
                    glm::vec3(tri.v1().x, ceiling_height, tri.v1().y),
                    glm::vec3(tri.v2().x, ceiling_height, tri.v2().y),
                    DeviceMaterial(ceiling_texture)));
        }

        sector_geometry.ceiling.insert(sector_geometry.ceiling.end(), ceiling_triangles.begin(),
                                       ceiling_triangles.end());
        scene_data.triangles.insert(scene_data.triangles.end(), ceiling_triangles.begin(), ceiling_triangles.end());
    }
}

DeviceMaterial get_device_material(short texture_number,
                                   wad::Wad &wad,
                                   wad::GraphicsData &graphics_data) {
    auto texture = get_device_texture(texture_number, wad, graphics_data);
    const auto &tex = graphics_data.get_texture(texture_number);

    return RT_GetMaterial(tex.name(), texture);
}

DeviceTexture *get_device_texture(short texture_number,
                                  wad::Wad &wad,
                                  wad::GraphicsData &graphics_data) {
    auto texture = detail::texture_cache.get_texture(texture_number);
    if (texture) {
        return texture;
    }

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

    return create_device_type<DeviceTexture>(pixels, 64, 64);
}

void RT_ChangeSideTopTexture(side_t *side, int texture_num) {
    detail::sidedef_to_wall_lookup[side].top->material = DeviceMaterial(get_device_texture(texture_num, *detail::wad,
                                                                           *detail::graphics_data));
}

void RT_ChangeSideMidTexture(side_t *side, int texture_num) {
    detail::sidedef_to_wall_lookup[side].middle->material = DeviceMaterial(get_device_texture(texture_num, *detail::wad,
                                                                              *detail::graphics_data));
}

void RT_ChangeSideBottomTexture(side_t *side, int texture_num) {
    detail::sidedef_to_wall_lookup[side].bottom->material = DeviceMaterial(get_device_texture(texture_num, *detail::wad,
                                                                              *detail::graphics_data));
}