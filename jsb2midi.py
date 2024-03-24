import numpy as np
from midiutil import MIDIFile
import torch
from pprint import pprint

# starts with 35 35 
base_file = '71 68 64 40 71 68 64 40 71 68 64 40 71 68 64 40 71 68 62 40 69 64 61 45 69 64 61 45 69 64 61 45 69 64 61 45 73 68 61 53 73 68 61 53 73 68 61 53 73 68 61 53 73 68 61 53 69 66 61 54 69 66 61 54 69 66 61 54 69 66 61 54 71 66 61 54 71 66 62 47 71 66 62 47 71 66 62 47 71 66 62 47 71 66 62 47 69 66 62 47 69 66 62 47 76 69 64 49 76 69 64 49 76 69 64 49 76 69 64 49 76 69 64 49 76 69 64 49 78 71 62 47 78 71 62 47 78 69 62 61 78 69 62 61 78 71 62 59 78 71 62 59 78 71 62 59 78 71 62 57 79 71 62 57 78 71 62 55 78 71 62 55 76 64 62 57 76 64 62 57 74 66 62 57 74 66 62 57 76 67 61 45 76 67 61 45 76 67 61 45 76 67 61 45 76 67 61 45 74 66 57 50 74 66 57 50 74 66 57 50 74 66 57 50 74 66 57 50 74 66 57 50 74 66 57 50 74 66 57 50 74 66 57 50 74 66 69 50 74 66 69 50 74 66 69 50 74 66 69 57 74 66 69 57 74 66 69 57 76 66 69 57 76 66 69 45 76 66 69 45 76 66 69 45 76 67 62 47 76 67 62 47 79 69 62 47 79 69 62 47 79 69 62 47 79 71 62 47 79 71 64 49 79 71 64 49 79 71 64 49 79 71 64 49 79 71 64 49 78 71 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 76 69 61 57 76 69 61 57 76 69 61 57 76 69 61 45 76 69 61 45 76 69 61 45 76 69 61 45 76 69 61 45 74 69 66 50 74 69 66 50 74 69 66 50 74 69 66 50 74 69 66 50 74 69 66 50 74 69 66 50 78 69 62 50 78 69 62 50 78 69 62 50 78 69 62 50 76 69 61 57 76 69 61 57 76 69 61 57 76 69 61 57 76 68 61 57 76 68 61 57 76 70 61 55 76 70 61 55 76 70 61 54 76 70 61 54 76 70 61 54 74 70 61 47 74 66 61 47 74 66 61 47 74 66 59 47 74 66 59 47 74 66 59 47 74 66 59 47 74 66 59 47 74 66 59 47 74 66 59 47 74 66 59 49 74 66 59 49 74 66 59 50 74 66 59 50 74 66 59 52 74 66 59 52 73 66 59 54 73 66 59 54 73 66 59 54 73 66 59 54 73 66 57 54 73 66 57 54 73 66 57 52 73 66 57 52 71 66 57 50 71 66 57 50 71 66 57 47 71 66 57 47 71 66 57 47 71 64 56 52 71 64 56 52 71 62 56 52 71 62 56 52 69 61 52 45 69 61 52 45 69 61 52 45 69 61 52 45 69 61 52 45 69 61 52 45 69 61 52 45 69 61 52 45 71 62 55 43 71 62 55 43 71 62 55 43 71 62 55 43 69 62 54 50 69 62 54 50 69 62 54 50 69 62 54 50 69 62 54 50 69 62 54 50 69 62 54 50 69 62 54 50 67 64 59 52 67 64 59 52 69 64 60 52 69 64 60 52 69 64 60 52 69 64 60 52 71 66 54 51 71 66 54 51 71 66 54 51 71 66 54 51 67 64 59 40 67 64 59 40 67 64 59 40 67 64 59 40 67 64 59 40 67 64 59 40 67 64 59 40 67 64 59 40 67 64 59 40 67 64 59 40 71 64 59 43 71 64 59 43 71 64 59 43 71 64 59 43 69 64 60 45 69 64 60 45 69 64 60 45 69 64 60 45 66 63 54 45 66 63 54 45 66 62 54 50 66 62 54 50 67 64 55 52 67 64 55 52 67 64 55 52 67 64 55 52 69 57 62 54 69 57 62 54 69 57 61 54 69 57 61 54 71 62 59 55'

number_list = [int(i) for i in base_file.split(' ')]

new_list = []

for i in range(len(number_list) // 4):
    new_list.append(number_list[i * 4: i * 4 + 4])

file = np.array(new_list, dtype=int)

tempo = 120
time_signature = (4, 4)

duration = 1 / 4

midi = MIDIFile(4)

for i in range(4):
    midi.addTempo(i, 0, tempo)

# print(np.array(data['train'][1]).T.shape)

prev_pitch = [None] * 4
duration_count = [0] * 4

# Add each pitch as a note on the track
for i, pitches_at_time in enumerate(file):
    # Calculate the time for the note (in 16th notes)
    time = i * duration

    # Add each pitch to its corresponding track
    for track, pitch in enumerate(pitches_at_time):
        # If the current pitch is the same as the previous pitch, increment the duration count
        if pitch == prev_pitch[track]:
            duration_count[track] += 1
        # Otherwise, add a new note to the track with the previous pitch and duration
        else:
            if prev_pitch[track] is not None:
                midi.addNote(
                    track,
                    0,
                    prev_pitch[track],
                    time - duration * duration_count[track],
                    duration * duration_count[track],
                    100,
                )
            duration_count[track] = 1
            prev_pitch[track] = pitch

# Add the last note to each track
for track in range(4):
    if prev_pitch[track] is not None:
        midi.addNote(
            track,
            0,
            prev_pitch[track],
            time + duration - duration * duration_count[track],
            duration * duration_count[track],
            100,
        )

# Write the MIDI data to a file
with open(f"jop.mid", "wb") as midi_file:
    try:
        midi.writeFile(midi_file)
    except Exception:
        print("???")
