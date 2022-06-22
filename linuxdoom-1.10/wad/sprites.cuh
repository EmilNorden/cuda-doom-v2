#ifndef SPRITES_CUH_
#define SPRITES_CUH_

#include "wad.cuh"
#include <string>
#include <iostream>


//
// Frame flags:
// handles maximum brightness (torches, muzzle flare, light sources)
//
#define FF_FULLBRIGHT	0x8000	// flag in thing->frame
#define FF_FRAMEMASK	0x7fff

namespace wad {

    struct SpriteFrame {
        bool rotate;
        std::array<int, 8> lumps;
        std::array<bool, 8> flip; // Might not be needed
    };

    struct Sprite {
        std::vector<SpriteFrame> frames;
    };

    class SpriteData {
    public:
        SpriteData(Wad &wad, char **sprite_names, int sprite_count) {
            m_sprite_lumps_start = wad.get_lump_number("S_START").value();
            auto sprite_lumps_end = wad.get_lump_number("S_END").value();

            for (int i = 0; i < sprite_count; ++i) {
                auto sprname = sprite_names[i];
                std::array<SpriteFrame, 29> sprite_frames{};
                int max_frame = -1;

                for (auto lump_number = m_sprite_lumps_start + 1; lump_number < sprite_lumps_end; ++lump_number) {
                    auto lump_name = wad.get_lump_name(lump_number);

                    if (lump_name.rfind(sprname, 0) != 0) {
                        continue;
                    }

                    auto lump = wad.get_lump(lump_number);

                    auto frame_number = lump.name[4] - 'A';
                    auto rotation = lump.name[5] - '0';

                    if (frame_number >= 29 || rotation > 8) {
                        std::cerr << "Something fishy is going on\n";
                        exit(1);
                    }

                    update_frame(sprite_frames, frame_number, rotation, m_sprite_lumps_start, lump, false);

                    if (frame_number > max_frame) {
                        max_frame = frame_number;
                    }

                    if (lump.name.length() > 6) {
                        frame_number = lump.name[6] -
                                       'A'; // Not sure why the frame is given again. From what I've seen the frame number never changes for flipped sprites
                        rotation = lump.name[7] - '0';
                        update_frame(sprite_frames, frame_number, rotation, m_sprite_lumps_start, lump, true);

                        if (frame_number > max_frame) {
                            max_frame = frame_number;
                        }
                    }
                }

                m_sprites.push_back(Sprite{
                        std::vector<SpriteFrame>(sprite_frames.begin(), sprite_frames.begin() + max_frame + 1)
                });
            }
        }

        [[nodiscard]] const std::vector<Sprite>& sprites() const { return m_sprites; }
        [[nodiscard]] int sprite_lumps_start() const { return m_sprite_lumps_start; }

    private:
        std::vector<Sprite> m_sprites;
        int m_sprite_lumps_start;

        void
        update_frame(std::array<SpriteFrame, 29> &sprite_frames, int frame_number, int rotation, int first_sprite_lump,
                     Lump &lump, bool flipped) {
            auto current_frame = &sprite_frames[frame_number];
            if (rotation == 0) {
                auto lump_number = lump.number - first_sprite_lump;
                current_frame->rotate = false;
                for (int i = 0; i < 8; ++i) {
                    current_frame->lumps[i] = lump_number;
                    current_frame->flip[i] = flipped;
                }
                return;
            }

            current_frame->rotate = true;
            rotation--;

            current_frame->lumps[rotation] = lump.number - first_sprite_lump;
            current_frame->flip[rotation] = flipped;
        }
    };

}
#endif