#include "graphics_data.cuh"
#include "lump_reader.cuh"

namespace wad {
    Picture read_picture(Lump &lump) {
        wad::LumpReader reader(lump);
        auto patch_width = reader.read_i16();

        std::uint16_t patch_height = reader.read_i16();

        std::uint16_t patch_leftoffset = reader.read_i16();

        std::uint16_t patch_topoffset = reader.read_i16();

        std::vector<std::int32_t> columnofs;
        columnofs.resize(patch_width);
        for (int i = 0; i < patch_width; ++i) {
            columnofs[i] = reader.read_i32();
        }

        std::vector<std::uint16_t> pixels(patch_width * patch_height, 0xFFFF);
        for (auto i = 0; i < patch_width; ++i) {
            // file.seekg(patch_lump.offset + columnofs[i]);
            reader.seek(columnofs[i]);

            std::uint8_t topdelta = 0x0;
            while (topdelta != 0xFF) {
                topdelta = reader.read_u8();

                if (topdelta == 0xFF) {
                    break;
                }

                auto length = reader.read_u8();

                auto dummy = reader.read_u8();

                for (auto j = 0; j < length; ++j) {
                    auto palette_index = reader.read_u8();

                    auto y = j + topdelta;
                    auto x = i;
                    auto pixel_index = (y * patch_width) + x;
                    if (pixel_index >= 0 && pixel_index < pixels.size()) {
                        pixels[pixel_index] = palette_index;
                    }
                }

                dummy = reader.read_u8();
            }
        }

        return wad::Picture{lump.name, pixels, patch_width, patch_height, patch_topoffset, patch_leftoffset};
    }

    Picture flip_picture(const Picture& picture) {
        std::vector<std::uint16_t> flipped_pixels;
        flipped_pixels.resize(picture.pixels.size());
        for(int y = 0; y < picture.height; ++y) {
            for(int x = 0; x < picture.width; ++x) {
                auto source_index = (y * picture.width) + x;
                auto destination_index = (y * picture.width) + ((picture.width - x) - 1);
                flipped_pixels[destination_index] = picture.pixels[source_index];
            }
        }

        return Picture{picture.name, flipped_pixels, picture.width, picture.height, picture.top_offset, picture.left_offset};
    }
}

std::vector<wad::MapTexture> read_textures(wad::Wad &wad) {
    auto texture1 = wad.get_lump("TEXTURE1");
    wad::LumpReader reader(texture1);

    auto numtextures = reader.read_i32();

    std::vector<int> offsets;
    offsets.reserve(numtextures);
    for (int i = 0; i < numtextures; ++i) {
        offsets.push_back(reader.read_i32());
    }

    std::vector<wad::MapTexture> textures;
    for (int i = 0; i < numtextures; ++i) {
        auto offset = offsets[i];
        reader.seek(offset);

        auto name = reader.read_fixed_length_string<8>();
        auto masked = reader.read_i32();
        auto width = reader.read_i16();
        auto height = reader.read_i16();
        auto column_directory = reader.read_i32();

        auto patch_count = reader.read_i16();

        std::vector<wad::MapTexturePatch> patches;
        for (int j = 0; j < patch_count; ++j) {
            wad::MapTexturePatch patch{};
            patch.originx = reader.read_i16();
            patch.originy = reader.read_i16();
            patch.patch = reader.read_i16();
            patch.stepdir = reader.read_i16();
            patch.colormap = reader.read_i16();

            patches.push_back(patch);
        }

        textures.emplace_back(name, masked, width, height, column_directory, patches);
    }

    return textures;
}

std::vector<wad::Picture> read_sprites(wad::Wad &wad) {
    auto sprite_lumps_start = wad.get_lump_number("S_START").value();
    auto sprite_lumps_end = wad.get_lump_number("S_END").value();

    std::vector<wad::Picture> sprites;
    for (int i = sprite_lumps_start + 1; i < sprite_lumps_end; ++i) {
        auto sprite_lump = wad.get_lump(i);

        sprites.push_back(wad::read_picture(sprite_lump));
    }

    return sprites;
}

std::vector<std::string> read_patch_names(wad::Wad &wad) {
    auto lump = wad.get_lump("PNAMES");
    wad::LumpReader reader(lump);

    auto nummappatches = reader.read_i32();

    std::vector<std::string> patch_names;
    for (int i = 0; i < nummappatches; ++i) {
        patch_names.push_back(reader.read_fixed_length_string<8>());
    }

    return patch_names;
}

std::array<uint8_t, 14 * 768> read_palette(wad::Wad &wad) {
    auto lump = wad.get_lump("PLAYPAL");
    std::array<uint8_t, 14 * 768> palette{};

    std::copy(lump.data.begin(), lump.data.end(), palette.begin());

    return palette;

}

namespace wad {
    GraphicsData::GraphicsData(Wad &wad) {
        m_textures = read_textures(wad);
        m_sprites = read_sprites(wad);
        m_patch_names = read_patch_names(wad);
        m_palette = read_palette(wad);
    }

    const MapTexture &GraphicsData::get_texture(std::string_view name) const {
        for (auto &texture: m_textures) {
            if (texture.name() == name) {
                return texture;
            }
        }

        throw std::runtime_error(fmt::format("Texture '{}' does not exist", name));
    }

    const MapTexture &GraphicsData::get_texture(short number) const {
        if(number >= m_textures.size()) {
            throw std::runtime_error(fmt::format("Texture at index '{}' does not exist", number));
        }

        return m_textures[number];
    }

    const Picture &GraphicsData::get_sprite(const std::string &name) const {
        for(auto &sprite : m_sprites) {
            if(sprite.name == name) {
                return sprite;
            }
        }

        throw std::runtime_error(fmt::format("Sprite '{}' does not exist", name));
    }

    std::vector<std::uint16_t> MapTexture::get_pixels(Wad &wad, const std::vector<std::string> &patch_names) const {
        auto pixels = std::vector<std::uint16_t>(m_width * m_height, 0xFFFF);
        for (auto &patch: m_patches) {
            const auto &patch_name = patch_names[patch.patch];

            auto patch_lump = wad.get_lump(patch_name);
            auto picture = read_picture(patch_lump);

            for (int patch_y = 0; patch_y < picture.height; ++patch_y) {
                for (int patch_x = 0; patch_x < picture.width; ++patch_x) {
                    auto dest_y = patch_y + patch.originy;
                    auto dest_x = patch_x + patch.originx;

                    auto pixel_index = (dest_y * m_width) + dest_x;

                    auto palette_index = picture.pixels[(patch_y * picture.width) + patch_x];
                    if (pixel_index >= 0 && pixel_index < pixels.size() && palette_index != 0xFFFF) {
                        pixels[pixel_index] = palette_index;
                    }
                }
            }
        }

        return pixels;
    }
}