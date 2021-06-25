
#ifndef DOOM_MUS_PARSER_H
#define DOOM_MUS_PARSER_H

#include <stdint.h>

typedef enum {
    release_note = 0,
    play_note = 1,
    pitch_bend = 2,
    system_event = 3,
    change_controller = 4,
    score_end = 6
} mus_event_type;

typedef struct mus_event_release_note {
    uint8_t note;
} mus_event_release_note_t;

typedef struct mus_event_play_note {
    uint8_t note;
    uint8_t volume;
} mus_event_play_note_t;

typedef struct mus_event_pitch_bend {
    uint8_t value;
} mus_event_pitch_bend_t;

typedef struct mus_event_system {
    uint8_t number; // TODO: Create enum?
} mus_event_system_t;

typedef struct mus_event_change_controller {
    uint8_t number;
    uint8_t value;
} mus_event_change_controller_t;

typedef struct mus_event {
    mus_event_type type;
    uint32_t ticks_before_next_event;
    uint8_t channel;
    mus_event_release_note_t *release_note;
    mus_event_play_note_t *play_note;
    mus_event_pitch_bend_t *pitch_bend;
    mus_event_system_t *system;
    mus_event_change_controller_t *controller;
} mus_event_t;

typedef struct mus_file {
    uint16_t score_length;
    uint16_t score_start;
    uint16_t primary_channel_count;
    uint16_t secondary_channel_count;
    uint16_t instrument_count;
    uint16_t *instruments;

    uint16_t event_count;
    mus_event_t *events;
} mus_file_t;

mus_file_t* mus_parse(void* mem, int length);

void mus_free(mus_file_t* mus);


#endif //DOOM_MUS_PARSER_H
