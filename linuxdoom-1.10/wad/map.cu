#include "map.cuh"
#include <fmt/core.h>
#include "lump_reader.cuh"

std::vector<wad::Thing> read_things(const wad::Lump &lump) {
    wad::LumpReader reader(lump);

    std::vector<wad::Thing> things;
    while(!reader.end_of_lump()) {
        wad::Thing thing{};
        thing.x = reader.read_i16();
        thing.y = reader.read_i16();
        thing.angle = reader.read_i16();
        thing.type = reader.read_i16();
        thing.flags = reader.read_i16();
        things.push_back(thing);
    }
    return things;
}

std::vector<wad::LineDef> read_linedefs(const wad::Lump &lump) {
    wad::LumpReader reader(lump);

    std::vector<wad::LineDef> linedefs;
    while(!reader.end_of_lump()) {
        wad::LineDef line{};
        line.start_vertex = reader.read_i16();
        line.end_vertex = reader.read_i16();
        line.flags = reader.read_i16();
        line.special_type = reader.read_i16();
        line.sector_tag = reader.read_i16();
        line.right_sidedef = reader.read_i16();
        line.left_sidedef = reader.read_i16();
        linedefs.push_back(line);
    }
    return linedefs;
}

std::vector<wad::SideDef> read_sidedefs(const wad::Lump &lump) {
    wad::LumpReader reader(lump);

    std::vector<wad::SideDef> sidedefs;
    while(!reader.end_of_lump()) {
        wad::SideDef side{};

        side.xoffset = reader.read_i16();
        side.yoffset = reader.read_i16();

        side.upper_texture = reader.read_fixed_length_string<8>();
        side.lower_texture = reader.read_fixed_length_string<8>();
        side.middle_texture = reader.read_fixed_length_string<8>();

        side.sector = reader.read_i16();

        sidedefs.push_back(side);
    }

    return sidedefs;
}

std::vector<wad::Vertex> read_vertices(const wad::Lump &lump) {
    wad::LumpReader reader(lump);

    std::vector<wad::Vertex> vertices;
    while(!reader.end_of_lump()) {
        wad::Vertex vertex{};

        vertex.x = reader.read_i16();
        vertex.y = reader.read_i16();
        vertices.push_back(vertex);
    }
    return vertices;
}

std::vector<wad::Sector> read_sectors(const wad::Lump &lump) {
    wad::LumpReader reader(lump);

    std::vector<wad::Sector> sectors;
    while(!reader.end_of_lump()) {
        wad::Sector sector{};

        sector.floor_height = reader.read_i16();
        sector.ceiling_height = reader.read_i16();

        sector.floor_texture = reader.read_fixed_length_string<8>();
        sector.ceiling_texture = reader.read_fixed_length_string<8>();

        sector.light_level = reader.read_i16();
        sector.type = reader.read_i16();
        sector.tag = reader.read_i16();

        sectors.push_back(sector);
    }

    return sectors;
}

wad::MapData::MapData(wad::Wad &wad, int map_number) {
    auto map_name = fmt::format("MAP{:02}", map_number);
    auto map_lump = wad.get_lump(map_name);

    m_things = read_things(wad.get_lump(map_lump.number + 1));
    m_linedefs = read_linedefs(wad.get_lump(map_lump.number + 2));
    m_sidedefs = read_sidedefs(wad.get_lump(map_lump.number + 3));
    m_vertices = read_vertices(wad.get_lump(map_lump.number + 4));
    m_sectors = read_sectors(wad.get_lump(map_lump.number + 8));
}
