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
    impl::DirectoryEntry read_directory_entry(std::ifstream &file, int additional_offset) {
        int offset;
        file.read(reinterpret_cast<char *>(&offset), sizeof(int));

        int size;
        file.read(reinterpret_cast<char *>(&size), sizeof(int));

        auto name = read_fixed_length_string<8>(file);

        return impl::DirectoryEntry{
                offset + additional_offset,
                size,
                name
        };
    }

    Wad::Wad(const std::vector<std::filesystem::path> &paths) {
        for (const auto &path: paths) {
            std::ifstream file(path, std::ios::in | std::ios::binary);

            auto id = read_fixed_length_string<4>(file);

            if (id != "PWAD" && id != "IWAD") {
                std::cerr << "WAD does not contain PWAD or IWAD header. Things might not work properly." << std::endl;;
            }

            int numlumps;
            file.read(reinterpret_cast<char *>(&numlumps), sizeof(int));

            int infotableofs;
            file.read(reinterpret_cast<char *>(&infotableofs), sizeof(int));

            // C++ I/O sucks.
            file.seekg(0, std::ios::end);
            auto file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            auto data_offset = m_data.size();
            m_data.reserve(m_data.size() + file_size);
            std::vector<uint8_t> contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

            m_data.insert(m_data.end(), contents.begin(), contents.end());

            file.seekg(infotableofs);

            for (int i = 0; i < numlumps; ++i) {
                m_directory.push_back(read_directory_entry(file, static_cast<int>(data_offset)));
            }
        }
    }

    std::optional<int> Wad::get_lump_number(std::string_view name) const {
        for (auto i = static_cast<int>(m_directory.size()-1); i >= 0 ; --i) {
            if (m_directory[i].name == name) {
                return i;
            }
        }
        return std::nullopt;
    }

    Lump Wad::get_lump(int number) {
        auto &entry = m_directory[number];

        Lump lump;
        lump.name = entry.name;
        lump.number = number;

        if (entry.size > 0) {
            // TODO: Change to span when using C++20
            lump.data.reserve(entry.size);
            lump.data.insert(lump.data.begin(), m_data.begin() + entry.offset, m_data.begin() + entry.offset + entry.size);
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
