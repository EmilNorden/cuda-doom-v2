#ifndef WAD_CUH_
#define WAD_CUH_

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <optional>
#include <functional>
#include "graphics_data.cuh"


namespace wad {
    namespace impl {
        struct DirectoryEntry {
            int offset;
            int size;
            std::string name;
        };
    }

    struct Lump {
        std::string name;
        int number;
        std::vector<std::uint8_t> data;
    };


    class Wad {
    public:
        explicit Wad(const std::vector<std::filesystem::path> &paths);

        [[nodiscard]] std::optional<int> get_lump_number(std::string_view name) const;

        [[nodiscard]] std::string get_lump_name(int number);
        [[nodiscard]] Lump get_lump(int number);
        [[nodiscard]] Lump get_lump(std::string_view name);

    private:
        std::vector<std::uint8_t> m_data;
        std::vector<impl::DirectoryEntry> m_directory;
    };
}


#endif