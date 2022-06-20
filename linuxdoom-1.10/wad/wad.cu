#include "wad.cuh"
#include <fstream>
#include <iostream>
#include <fmt/core.h>

template<size_t N>
std::string read_fixed_length_string(std::ifstream &file) {
    char str[N + 1];
    str[N] = 0;
    file.read(str, N);
    return std::string{str};
}

namespace wad {
    impl::DirectoryEntry read_directory_entry(std::ifstream &file) {
        int offset;
        file.read(reinterpret_cast<char *>(&offset), sizeof(int));

        int size;
        file.read(reinterpret_cast<char *>(&size), sizeof(int));

        auto name = read_fixed_length_string<8>(file);

        return impl::DirectoryEntry{
                offset,
                size,
                name
        };
    }

    Wad::Wad(std::filesystem::path path) {
        m_file = std::ifstream(path, std::ios::in | std::ios::binary);

        auto id = read_fixed_length_string<4>(m_file);

        if (id != "PWAD" && id != "IWAD") {
            std::cerr << "WAD does not contain PWAD or IWAD header. Things might not work properly." << std::endl;;
        }

        int numlumps;
        m_file.read(reinterpret_cast<char *>(&numlumps), sizeof(int));

        int infotableofs;
        m_file.read(reinterpret_cast<char *>(&infotableofs), sizeof(int));

        m_file.seekg(infotableofs);

        std::vector<impl::DirectoryEntry> directory;
        for (int i = 0; i < numlumps; ++i) {
            m_directory.push_back(read_directory_entry(m_file));
        }
    }

    std::optional<int> Wad::get_lump_number(std::string_view name) const {
        for (auto i = 0; i < m_directory.size(); ++i) {
            if (m_directory[i].name == name) {
                return i;
            }
        }
        return std::nullopt;
    }

    Lump Wad::get_lump(int number) {
        auto &entry = m_directory[number];

        m_file.seekg(entry.offset);
        Lump lump;
        lump.name = entry.name;
        lump.number = number;

        if (entry.size > 0) {
            lump.data.resize(entry.size);
            m_file.read(reinterpret_cast<char *>(lump.data.data()), entry.size);
        }


        return lump;
    }

    Lump Wad::get_lump(std::string_view name) {
        auto lump_number = get_lump_number(name);
        if (!lump_number.has_value()) {
            throw std::runtime_error(fmt::format("Unable to find lump '{}'", name));
        }

        return get_lump(lump_number.value());
    }

    std::string Wad::get_lump_name(int number) {
        auto &entry = m_directory[number];

        return entry.name;
    }
}
