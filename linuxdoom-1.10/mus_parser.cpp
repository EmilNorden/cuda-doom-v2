#include "mus_parser.h"
#include "doomtype.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define HEADER_MAGIC    0x1A53554D // 'MUS\x1A'

typedef struct binary_reader {
    void *read_ptr;
    int remaining_bytes;
} binary_reader_t;

boolean bin_read32(binary_reader_t *reader, uint32_t *val) {
    const int WORD_SIZE = 4;
    if (reader->remaining_bytes < WORD_SIZE) {
        return false;
    }

    uint32_t *read_ptr = (uint32_t *) reader->read_ptr;
    *val = *read_ptr++;

    reader->read_ptr = read_ptr;
    reader->remaining_bytes -= WORD_SIZE;
    return true;
}

boolean bin_read16(binary_reader_t *reader, uint16_t *val) {
    const int WORD_SIZE = 2;
    if (reader->remaining_bytes < WORD_SIZE) {
        return false;
    }

    uint16_t *read_ptr = (uint16_t *) reader->read_ptr;
    *val = *read_ptr++;

    reader->read_ptr = read_ptr;
    reader->remaining_bytes -= WORD_SIZE;
    return true;
}

boolean bin_read8(binary_reader_t *reader, uint8_t *val) {
    const int WORD_SIZE = 1;
    if (reader->remaining_bytes < WORD_SIZE) {
        return false;
    }

    uint8_t *read_ptr = (uint8_t *) reader->read_ptr;
    *val = *read_ptr++;

    reader->read_ptr = read_ptr;
    reader->remaining_bytes -= WORD_SIZE;
    return true;
}

boolean bin_has_more_data(binary_reader_t *reader) {
    return reader->remaining_bytes > 0;
}

boolean mus_read_time(binary_reader_t *reader, uint32_t *time) {
    uint32_t total_time = 0;

    uint8_t time_chunk;
    do {
        if (!bin_read8(reader, &time_chunk)) {
            return false;
        }
        total_time = total_time * 128 + (time_chunk & 0x7F);
    } while (time_chunk & 0x80);

    *time = total_time;
    return true;
}

mus_file_t *mus_parse(void *mem, int length) {
    mus_file_t *result = (mus_file_t*)malloc(sizeof(mus_file_t));

    binary_reader_t reader;
    reader.read_ptr = (byte *) mem;
    reader.remaining_bytes = length;

    uint32_t header_magic_value;
    if (!bin_read32(&reader, &header_magic_value)) {
        return NULL;
    }

    if (header_magic_value != HEADER_MAGIC) {
        fprintf(stderr, "MUS data: invalid header!");
        return NULL;
    }

    if (!bin_read16(&reader, &result->score_length)) {
        return NULL;
    }

    if (!bin_read16(&reader, &result->score_start)) {
        return NULL;
    }

    if (!bin_read16(&reader, &result->primary_channel_count)) {
        return NULL;
    }

    if (!bin_read16(&reader, &result->secondary_channel_count)) {
        return NULL;
    }

    if (!bin_read16(&reader, &result->instrument_count)) {
        return NULL;
    }

    uint16_t dummy_value;
    if (!bin_read16(&reader, &dummy_value)) {
        return NULL;
    }

    result->instruments = (uint16_t*)malloc(sizeof(uint16_t) * result->instrument_count);
    for (int i = 0; i < result->instrument_count; ++i) {
        if (!bin_read16(&reader, &result->instruments[i])) {
            return NULL;
        }
    }

    int event_capacity = 8;
    uint16_t event_count = 0;
    mus_event_t *events = (mus_event_t*)malloc(sizeof(mus_event_t) * event_capacity);

    uint8_t channel_volumes[16];
    memset(channel_volumes, 0, 16);
    while (bin_has_more_data(&reader)) {
        if (event_count == event_capacity) {
            event_capacity *= 2;
            events = (mus_event_t*)realloc(events, sizeof(mus_event_t) * event_capacity);
        }

        byte event_descriptor;
        if (!bin_read8(&reader, &event_descriptor)) {
            return NULL;
        }

        mus_event_t *event = &events[event_count];
        memset(event, 0, sizeof(mus_event_t));
        event->type = static_cast<mus_event_type>((event_descriptor & 0b01110000) >> 4);
        event->channel = event_descriptor & 0b00001111;


        if (event->channel >= 16) {
            fprintf(stderr, "MUS event contains invalid channel %d\n", event->channel);
            return NULL;
        }

        switch (event->type) {
            case release_note: {
                uint8_t number;
                if (!bin_read8(&reader, &number)) {
                    return NULL;
                }

                event->release_note = (mus_event_release_note_t*)malloc(sizeof(mus_event_release_note_t));
                event->release_note->note = number;

                break;
            }
            case play_note: {
                uint8_t number;
                if (!bin_read8(&reader, &number)) {
                    return NULL;
                }

                int has_volume = number & 0x80;
                uint8_t volume;
                if (has_volume) {
                    if (!bin_read8(&reader, &volume)) {
                        return NULL;
                    }
                    volume = channel_volumes[event->channel] = volume;
                } else {
                    volume = channel_volumes[event->channel];
                }

                event->play_note = (mus_event_play_note_t*)malloc(sizeof(mus_event_play_note_t));
                event->play_note->note = number & ~0x80;
                event->play_note->volume = volume;
                break;
            }
            case pitch_bend: {
                uint8_t value;
                if (!bin_read8(&reader, &value)) {
                    return NULL;
                }

                event->pitch_bend = (mus_event_pitch_bend_t*)malloc(sizeof(mus_event_pitch_bend_t));
                event->pitch_bend->value = value;
                break;
            }
            case system_event: {
                uint8_t number;
                if (!bin_read8(&reader, &number)) {
                    return NULL;
                }

                if (number < 10 || number > 14) {
                    fprintf(stderr, "Invalid system event: %d\n", number);
                    return NULL;
                }

                event->system = (mus_event_system_t*)malloc(sizeof(mus_event_system_t));
                event->system->number = number;
                break;
            }
            case change_controller: {
                uint8_t number;
                if (!bin_read8(&reader, &number)) {
                    return NULL;
                }

                uint8_t value;
                if (!bin_read8(&reader, &value)) {
                    return NULL;
                }


                if (number > 9) {
                    fprintf(stderr, "Invalid change controller number: %d\n", number);
                    return NULL;
                }

                if(number == 3 && event->channel == 0 && value == 0x7e) {
                   int foo = 3421;
                }

                event->controller = (mus_event_change_controller_t*)malloc(sizeof(mus_event_change_controller_t));
                event->controller->number = number;
                event->controller->value = value;
                break;
            }
            case score_end:
                break;
            default:
                fprintf(stderr, "Invalid event type: %d\n", events[event_count].type);
                return NULL;
        }

        int is_last_in_group = event_descriptor & 0b10000000;

        events[event_count].ticks_before_next_event = 0;
        if (is_last_in_group) {
            if (!mus_read_time(&reader, &events[event_count].ticks_before_next_event)) {
                return NULL;
            }
        }

        event_count++;
    }

    result->events = events;
    result->event_count = event_count - 1;

    /*for (int i = 0; i < result->event_count; ++i) {
        printf("%d\t%d\t%d\t%d\n",
               result->events[i].type,
               result->events[i].channel,
               result->events[i].number,
               result->events[i].value);
    }

    fflush(stdout);
*/
    return result;
}

void mus_free(mus_file_t* mus) {
    if(!mus) {
        return;
    }

    free(mus->instruments);
    mus->instruments = NULL;

    for(int i = 0; i < mus->event_count; ++i) {
        mus_event_t *event = &mus->events[i];
        switch(event->type) {
            case release_note:
                free(event->release_note);
                event->release_note = NULL;
                break;
            case play_note:
                free(event->play_note);
                event->play_note = NULL;
                break;
            case pitch_bend:
                free(event->pitch_bend);
                event->pitch_bend = NULL;
                break;
            case system_event:
                free(event->system);
                event->system = NULL;
                break;
            case change_controller:
                free(event->controller);
                event->controller = NULL;
                break;
            case score_end:
                /* No payload for this event */
                break;
        }
    }

    free(mus->events);
    mus->events = NULL;
    free(mus);
}