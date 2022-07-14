
#ifndef DOOM_MIDI_CONVERTER_H
#define DOOM_MIDI_CONVERTER_H

#include "doomtype.h"

struct mus_file;

typedef struct midi_data {
    void* data;
    int length;
} midi_data_t;

void convert_mus_to_midi(struct mus_file* mus, midi_data_t *out_midi);

bool is_midi(void *data, int length);

#endif //DOOM_MIDI_CONVERTER_H
