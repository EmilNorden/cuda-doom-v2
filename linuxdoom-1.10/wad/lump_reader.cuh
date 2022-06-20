#ifndef LUMP_READER_CUH_
#define LUMP_READER_CUH_

#include <cstdint>
#include "wad.cuh"
#include <fmt/core.h>

namespace wad {
    class LumpReader {
    public:
        explicit LumpReader(const Lump &lump) : m_lump(lump), m_position(0) {}

        [[nodiscard]] bool end_of_lump() const {
            return m_position == m_lump.data.size();
        }

        [[nodiscard]] std::uint8_t read_u8() {
            if (remaining_data() < sizeof(std::uint8_t)) {
                throw std::runtime_error(
                        fmt::format("Unable to read {} bytes from lump. Lump size is {} and current position is {}",
                                    sizeof(std::uint8_t), m_lump.data.size(), m_position));
            }

            auto byte = m_lump.data[m_position];
            m_position += sizeof(std::uint8_t);
            return byte;
        }

        [[nodiscard]] std::int16_t read_i16() {
            if (remaining_data() < sizeof(std::int16_t)) {
                throw std::runtime_error(
                        fmt::format("Unable to read {} bytes from lump. Lump size is {} and current position is {}",
                                    sizeof(std::int16_t), m_lump.data.size(), m_position));
            }

            auto lower = m_lump.data[m_position];
            auto higher = m_lump.data[m_position + 1];

            m_position += sizeof(std::int16_t);
            return static_cast<std::int16_t>(higher << 8) | lower;
        }

        [[nodiscard]] std::int32_t read_i32() {
            if (remaining_data() < sizeof(std::int32_t)) {
                throw std::runtime_error(
                        fmt::format("Unable to read {} bytes from lump. Lump size is {} and current position is {}",
                                    sizeof(std::int32_t), m_lump.data.size(), m_position));
            }

            auto byte1 = m_lump.data[m_position];
            auto byte2 = m_lump.data[m_position + 1];
            auto byte3 = m_lump.data[m_position + 2];
            auto byte4 = m_lump.data[m_position + 3];

            m_position += sizeof(std::int32_t);
            return (byte4 << 24) | (byte3 << 16) | (byte2 << 8) | byte1;
        }

        template<size_t N>
        std::string read_fixed_length_string() {
            if (remaining_data() < N) {
                throw std::runtime_error(
                        fmt::format("Unable to read {} bytes from lump. Lump size is {} and current position is {}",
                                    sizeof(std::int16_t), m_lump.data.size(), m_position));
            }

            char str[N + 1];
            str[N] = 0;

            for (int i = 0; i < N; ++i) {
                str[i] = m_lump.data[m_position + i];
            }

            m_position += N;
            return std::string{str};
        }

        void seek(size_t position) {
            if(position >= m_lump.data.size()) {
                throw std::runtime_error(fmt::format("Invalid call to seek: Position {} is invalid. Lump size is {}", position, m_lump.data.size()));
            }

            m_position = position;
        }

    private:

        [[nodiscard]] size_t remaining_data() const { return m_lump.data.size() - m_position; }

        const Lump &m_lump;
        size_t m_position;
    };
}


#endif