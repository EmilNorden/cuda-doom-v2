#include "midi_converter.h"
#include "mus_parser.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>

static inline uint16_t Reverse16(uint16_t value) {
    return (((value & 0x00FF) << 8) |
            ((value & 0xFF00) >> 8));
}

static inline uint32_t Reverse32(uint32_t value) {
    return (((value & 0x000000FF) << 24) |
            ((value & 0x0000FF00) << 8) |
            ((value & 0x00FF0000) >> 8) |
            ((value & 0xFF000000) >> 24));
}

typedef struct binary_writer {
    uint8_t *data;
    uint32_t capacity;
    uint32_t write_position;
} binary_writer_t;

binary_writer_t bw_new(uint32_t capacity) {
    binary_writer_t writer;
    writer.data = (uint8_t*)malloc(capacity);
    writer.capacity = capacity;
    writer.write_position = 0;

    return writer;
}

void bw_free(binary_writer_t *writer) {
    free(writer->data);
    writer->data = NULL;
}

static void bw_expand_capacity(binary_writer_t *writer) {
    writer->data = (uint8_t*)realloc(writer->data, writer->capacity * 2);
    writer->capacity = writer->capacity * 2;
}

static void bw_write_data(binary_writer_t *writer, uint8_t *data, uint32_t length) {
    while (writer->capacity - writer->write_position < length) {
        bw_expand_capacity(writer);
    }

    for (int i = 0; i < length; ++i) {
        writer->data[writer->write_position + i] = data[i];
    }
    writer->write_position += length;
}

static void bw_write32(binary_writer_t *writer, uint32_t value) {
    const int WORD_SIZE = 4;
    while (writer->capacity - writer->write_position < WORD_SIZE) {
        bw_expand_capacity(writer);
    }

    auto *write_ptr = &writer->data[writer->write_position];
    *write_ptr++ = value & 0xFF;
    *write_ptr++ = (value >> 8) & 0xFF;
    *write_ptr++ = (value >> 16) & 0xFF;
    *write_ptr++ = (value >> 24) & 0xFF;
    writer->write_position += WORD_SIZE;
}

static void bw_write16(binary_writer_t *writer, uint16_t value) {
    const int WORD_SIZE = 2;
    while (writer->capacity - writer->write_position < WORD_SIZE) {
        bw_expand_capacity(writer);
    }

    auto *write_ptr = &writer->data[writer->write_position];
    *write_ptr++ = value & 0xFF;
    *write_ptr++ = (value >> 8) & 0xFF;
    writer->write_position += WORD_SIZE;
}

static void bw_write8(binary_writer_t *writer, uint8_t value) {
    const int WORD_SIZE = 1;
    while (writer->capacity - writer->write_position < WORD_SIZE) {
        bw_expand_capacity(writer);
    }

    uint8_t *write_ptr = &writer->data[writer->write_position];
    *write_ptr = value;
    writer->write_position += WORD_SIZE;
}

static void write_header(binary_writer_t *writer) {
    const int MAGIC_HEADER = 0x6468544D;
    bw_write32(writer, MAGIC_HEADER);

    // Length
    bw_write32(writer, Reverse32(6));

    // Format
    bw_write16(writer, Reverse16(0));

    // Number of tracks
    bw_write16(writer, Reverse16(1));

    // Number of ticks per quarter note
    bw_write16(writer, Reverse16(70));
}

static void write_delay(uint32_t delay, binary_writer_t *track_writer) {
    uint32_t buffer = delay & 0x7f;
    while ((delay >>= 7) > 0) {
        buffer <<= 8;
        buffer |= 0x80;
        buffer += (delay & 0x7f);
    }

    while (1) {
        bw_write8(track_writer, (uint8_t)buffer);
        if (buffer & 0x80) buffer >>= 8;
        else
            break;
    }
}

static void write_event(binary_writer_t *writer, mus_event_t* event) {
    uint8_t channel = event->channel;
    if (channel == 0x0F) {
        channel = 0x09;
    }

    switch (event->type) {
        case release_note: {
            uint8_t midi_event_type = 0x80 + channel;
            bw_write8(writer, midi_event_type);
            bw_write8(writer, event->release_note->note);
            bw_write8(writer, 0x00); // How hard to release. Unknown?
            break;
        }
        case play_note: {
            uint8_t midi_event_type = 0x90 + channel;
            bw_write8(writer, midi_event_type);
            bw_write8(writer, event->play_note->note);
            bw_write8(writer, event->play_note->volume);
            break;
        }
        case pitch_bend: {
            uint8_t midi_event_type = 0xE0 + channel;
            bw_write8(writer, midi_event_type);
            // MUS uses the range 0-255, 128 being no bend
            // MIDI uses the range 0-16384, 8192 being no bend.
            // I need to transform the value to the new range
            auto bend = (uint16_t) (((float) event->pitch_bend->value / 128.0f) * 16384);
            bw_write16(writer, bend);
            break;
        }
        case change_controller: {
            switch (event->controller->number) {
                case 0: {
                    uint8_t midi_event_type = 0xC0 + channel;
                    bw_write8(writer, midi_event_type);
                    bw_write8(writer, event->controller->value);
                    break;
                }
                case 1: {
                    uint8_t midi_event_type = 0xB0 + channel;
                    bw_write8(writer, midi_event_type);
                    bw_write8(writer, 0x20); // bank select
                    bw_write8(writer, event->controller->value);
                    break;
                }
                case 2: {
                    uint8_t midi_event_type = 0xB0 + channel;
                    bw_write8(writer, midi_event_type);
                    bw_write8(writer, 0x01); // modulation
                    bw_write8(writer, event->controller->value); // modulation
                    break;
                }
                case 3: {
                    // volume
                    uint8_t midi_event_type = 0xB0 + channel;
                    bw_write8(writer, midi_event_type);
                    bw_write8(writer, 0x07); // volume change
                    bw_write8(writer, event->controller->value);
                    break;
                }
                case 4: {
                    // pan
                    uint8_t midi_event_type = 0xB0 + channel;
                    bw_write8(writer, midi_event_type);
                    bw_write8(writer, 0x0A); // pan
                    bw_write8(writer, event->controller->value);
                    break;
                }
                default: {
                    fprintf(stderr, "Unknown controller number: %d\n", event->controller->number);
                    exit(1);
                }
            }
            break;
        }
        default: {
            fprintf(stderr, "Unknown event type: %d\n", event->type);
            exit(1);
        }
    }

    write_delay(event->ticks_before_next_event, writer);
}

static boolean write_track(struct mus_file *mus, binary_writer_t *writer) {
    const int MAGIC_HEADER = 0x6B72544D;
    bw_write32(writer, MAGIC_HEADER);

    binary_writer_t track_writer = bw_new(512);

    write_delay(0, &track_writer);
    for (int i = 0; i < mus->event_count; ++i) {
        mus_event_t *event = &mus->events[i];
        write_event(&track_writer, event);
    }

    // write end event (meta event)
    bw_write8(&track_writer, 0xFF);
    bw_write8(&track_writer, 0x2F);
    bw_write8(&track_writer, 0x00);

    bw_write32(writer, Reverse32(track_writer.write_position));
    bw_write_data(writer, track_writer.data, track_writer.write_position);
    bw_free(&track_writer);

    return true;
}

void convert_mus_to_midi(struct mus_file *mus, midi_data_t *out_midi) {
    binary_writer_t writer = bw_new(512);
    write_header(&writer);
    write_track(mus, &writer);

    out_midi->data = writer.data;
    out_midi->length = writer.write_position;
}

bool is_midi(void *data, int length) {
    if(length < 4) {
        return false;
    }

    auto *header = reinterpret_cast<std::int32_t*>(data);
    return *header == 0x6468544d;
}