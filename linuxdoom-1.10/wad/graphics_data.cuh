#ifndef GRAPHICS_DATA_CUH_
#define GRAPHICS_DATA_CUH_

#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include <array>

namespace wad {
    class Wad;
    class Lump;

    struct Picture {
        std::string name;
        std::vector<std::uint16_t> pixels;
        int width;
        int height;
        std::uint16_t top_offset;
        std::uint16_t left_offset;
    };

    Picture read_picture(Lump& lump);
    Picture flip_picture(const Picture& picture);

    struct MapTexturePatch {
        std::int16_t originx;
        std::int16_t originy;
        std::int16_t patch;
        std::int16_t stepdir;
        std::int16_t colormap;
    };

    class MapTexture {
    public:
        MapTexture(std::string  name, std::int32_t masked, std::int16_t width, std::int16_t height, std::int32_t column_directory, std::vector<MapTexturePatch> patches)
            : m_name(std::move(name)), m_masked(masked), m_width(width), m_height(height), m_column_directory(column_directory), m_patches(std::move(patches)) {

        }

        std::vector<std::uint16_t> get_pixels(Wad& wad, const std::vector<std::string> &patch_names) const;

        [[nodiscard]] const std::string &name() const { return m_name; }
        [[nodiscard]] std::int16_t width() const { return m_width; }
        [[nodiscard]] std::int16_t height() const { return m_height; }
    private:
        std::string m_name;
        std::int32_t m_masked;
        std::int16_t m_width;
        std::int16_t m_height;
        std::int32_t m_column_directory;
        std::vector<MapTexturePatch> m_patches;
    };

    class GraphicsData {
    public:
        explicit GraphicsData(Wad& wad);

        [[nodiscard]] const std::vector<MapTexture>& textures() const { return m_textures; }
        [[nodiscard]] const std::vector<std::string>& patch_names() const { return m_patch_names; }
        [[nodiscard]] const std::array<std::uint8_t, 14*768>& palette() const { return m_palette; }

        [[nodiscard]] const MapTexture &get_texture(std::string_view name) const;
        [[nodiscard]] const MapTexture &get_texture(short number) const;

        [[nodiscard]] const Picture& get_sprite(const std::string &name) const;

    private:
        std::vector<MapTexture> m_textures;
        std::vector<Picture> m_sprites;
        std::vector<std::string> m_patch_names;
        std::array<std::uint8_t, 14*768> m_palette;
    };
}


#endif