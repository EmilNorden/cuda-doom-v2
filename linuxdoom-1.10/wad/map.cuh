#ifndef MAP_CUH_
#define MAP_CUH_

#include <cstdint>
#include <vector>
#include <string>
#include <iosfwd>
#include "wad.cuh"

namespace wad {
    struct Thing {
        std::int16_t x;
        std::int16_t y;
        std::int16_t angle;
        std::int16_t type;
        std::int16_t flags;
    };

    struct LineDef {
        std::int16_t start_vertex;
        std::int16_t end_vertex;
        std::int16_t flags;
        std::int16_t special_type;
        std::int16_t sector_tag;
        std::int16_t right_sidedef;
        std::int16_t left_sidedef;
    };

    struct SideDef {
        std::int16_t xoffset;
        std::int16_t yoffset;
        std::string upper_texture;
        std::string lower_texture;
        std::string middle_texture;
        std::int16_t sector;
    };

    struct Vertex {
        std::int16_t x;
        std::int16_t y;
    };

    struct Sector {
        std::int16_t floor_height;
        std::int16_t ceiling_height;
        std::string floor_texture;
        std::string ceiling_texture;
        std::int16_t light_level;
        std::int16_t type;
        std::int16_t tag;
    };

    class MapData {
        //dc
    public:
        MapData(Wad &wad, int map_number);

        [[nodiscard]] const std::vector<Thing>& things() const { return m_things; }
        [[nodiscard]] const std::vector<LineDef>& linedefs() const { return m_linedefs; }
        [[nodiscard]] const std::vector<SideDef>& sidedefs() const { return m_sidedefs; }
        [[nodiscard]] const std::vector<Sector>& sectors() const { return m_sectors; }
        [[nodiscard]] const std::vector<Vertex>& vertices() const { return m_vertices; }

    private:
        std::vector<Thing> m_things;
        std::vector<LineDef> m_linedefs;
        std::vector<SideDef> m_sidedefs;
        std::vector<Sector> m_sectors;
        std::vector<Vertex> m_vertices;
    };
}


#endif