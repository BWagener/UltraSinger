"""Midi creator module"""
import copy
import math
import os
from collections import Counter

import librosa
import numpy as np
import pretty_midi
import unidecode

from modules.Midi.MidiSegment import MidiSegment
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.console_colors import (
    ULTRASINGER_HEAD, blue_highlighted,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence

SYLLABLE_SEGMENT_SIZE = 0.1
SYLLABLE_SEGMENT_MAX_GAP_FOR_MERGE = 0.1

def create_midi_instrument(midi_segments: list[MidiSegment]) -> object:
    """Converts an Ultrastar data to a midi instrument"""

    print(f"{ULTRASINGER_HEAD} Creating midi instrument")

    instrument = pretty_midi.Instrument(program=0, name="Vocals")
    velocity = 100

    for i, midi_segment in enumerate(midi_segments):
        note = pretty_midi.Note(velocity, librosa.note_to_midi(midi_segment.note), midi_segment.start, midi_segment.end)
        instrument.notes.append(note)

    return instrument

def sanitize_for_midi(text):
    """
    Sanitize text for MIDI compatibility.
    Uses unidecode to approximate characters to ASCII.
    """
    return unidecode.unidecode(text)

def __create_midi(instruments: list[object], bpm: float, midi_output: str, midi_segments: list[MidiSegment]) -> None:
    """Write instruments to midi file"""

    print(f"{ULTRASINGER_HEAD} Creating midi file -> {midi_output}")

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for i, midi_segment in enumerate(midi_segments):
        sanitized_word = sanitize_for_midi(midi_segment.word)
        midi_data.lyrics.append(pretty_midi.Lyric(text=sanitized_word, time=midi_segment.start))
    for instrument in instruments:
        midi_data.instruments.append(instrument)
    midi_data.write(midi_output)


class MidiCreator:
    """Docstring"""


def convert_frequencies_to_notes(frequency: [str]) -> list[list[str]]:
    """Converts frequencies to notes"""
    notes = []
    for freq in frequency:
        notes.append(librosa.hz_to_note(float(freq)))
    return notes


def most_frequent(array: [str]) -> list[tuple[str, int]]:
    """Get most frequent item in array"""
    return Counter(array).most_common(1)


def find_nearest_index(array: list[float], value: float) -> int:
    """Nearest index in array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1

    return idx


def create_midi_notes_from_pitched_data(start_times: list[float], end_times: list[float], words: list[str], pitched_data: PitchedData) -> list[
    MidiSegment]:
    """Create midi notes from pitched data"""
    print(f"{ULTRASINGER_HEAD} Creating midi_segments")

    midi_segments = []

    for index, start_time in enumerate(start_times):
        end_time = end_times[index]
        word = str(words[index])

        midi_segment = create_midi_note_from_pitched_data(start_time, end_time, pitched_data, word)
        midi_segments.append(midi_segment)

        # todo: Progress?
        # print(filename + " f: " + str(mean))
    return midi_segments


def create_midi_note_from_pitched_data(start_time: float, end_time: float, pitched_data: PitchedData, word: str) -> MidiSegment:
    """Create midi note from pitched data"""

    start = find_nearest_index(pitched_data.times, start_time)
    end = find_nearest_index(pitched_data.times, end_time)

    if start == end:
        freqs = [pitched_data.frequencies[start]]
        confs = [pitched_data.confidence[start]]
    else:
        freqs = pitched_data.frequencies[start:end]
        confs = pitched_data.confidence[start:end]

    conf_f = get_frequencies_with_high_confidence(freqs, confs)

    notes = convert_frequencies_to_notes(conf_f)

    note = most_frequent(notes)[0][0]

    return MidiSegment(note, start_time, end_time, word)


def create_midi_segments_from_transcribed_data(transcribed_data: list[TranscribedData], pitched_data: PitchedData) -> tuple[list[MidiSegment], list[TranscribedData]]:
    start_times = []
    end_times = []
    words = []

    if transcribed_data:
        split_transcribed_data = split_syllables_into_segments(transcribed_data)

        for i, midi_segment in enumerate(split_transcribed_data):
            start_times.append(midi_segment.start)
            end_times.append(midi_segment.end)
            words.append(midi_segment.word)
        midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words,
                                                            pitched_data)

        merged_midi_segments, merged_transcribed_data = merge_syllable_segments(midi_segments, split_transcribed_data)
        return merged_midi_segments, merged_transcribed_data


def create_repitched_midi_segments_from_ultrastar_txt(pitched_data: PitchedData, ultrastar_txt: UltrastarTxtValue) -> list[MidiSegment]:
    start_times = []
    end_times = []
    words = []

    for i, note_lines in enumerate(ultrastar_txt.UltrastarNoteLines):
        start_times.append(note_lines.startTime)
        end_times.append(note_lines.endTime)
        words.append(note_lines.word)
    midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words, pitched_data)
    return midi_segments


def create_midi_file(
        real_bpm: float,
        song_output: str,
        midi_segments: list[MidiSegment],
        basename_without_ext: str,
) -> None:
    """Create midi file"""
    print(f"{ULTRASINGER_HEAD} Creating Midi with {blue_highlighted('pretty_midi')}")

    voice_instrument = [
        create_midi_instrument(midi_segments)
    ]

    midi_output = os.path.join(song_output, f"{basename_without_ext}.mid")
    __create_midi(voice_instrument, real_bpm, midi_output, midi_segments)


def split_syllables_into_segments(
        transcribed_data: list[TranscribedData],
) -> list[TranscribedData]:
    """Split every syllable into sub-segments"""
    segment_size_decimal_points = len(str(SYLLABLE_SEGMENT_SIZE).split(".")[1])
    new_data = []

    for i, data in enumerate(transcribed_data):
        duration = data.end - data.start
        if duration <= SYLLABLE_SEGMENT_SIZE:
            new_data.append(data)
            continue

        has_space = str(data.word).endswith(" ")
        first_segment = copy.deepcopy(data)
        filler_words_start = data.start + SYLLABLE_SEGMENT_SIZE
        remainder = data.end - filler_words_start
        first_segment.end = filler_words_start
        if has_space:
            first_segment.word = first_segment.word[:-1]

        new_data.append(first_segment)

        full_segments, partial_segment = divmod(remainder, SYLLABLE_SEGMENT_SIZE)

        if full_segments >= 1:
            for i in range(int(full_segments)):
                segment = TranscribedData()
                segment.word = "~"
                segment.start = filler_words_start + round(
                    i * SYLLABLE_SEGMENT_SIZE, segment_size_decimal_points
                )
                segment.end = segment.start + SYLLABLE_SEGMENT_SIZE
                new_data.append(segment)

        if partial_segment >= 0.01:
            segment = TranscribedData()
            segment.word = "~"
            segment.start = filler_words_start + round(
                full_segments * SYLLABLE_SEGMENT_SIZE, segment_size_decimal_points
            )
            segment.end = segment.start + partial_segment
            new_data.append(segment)

        if has_space:
            new_data[-1].word += " "
    return new_data


def merge_syllable_segments(
        midi_segments: list[MidiSegment], transcribed_data: list[TranscribedData]
) -> tuple[list[MidiSegment], list[TranscribedData]]:
    """Merge sub-segments of a syllable where the pitch is the same"""
    new_data = []
    new_midi_notes = []

    previous_data = None

    for i, data in enumerate(transcribed_data):
        if (
                str(data.word).startswith("~")
                and previous_data is not None
                and midi_segments[i].note == midi_segments[i - 1].note
                and data.start - previous_data.end <= SYLLABLE_SEGMENT_MAX_GAP_FOR_MERGE
        ):
            new_data[-1].end = data.end
            new_midi_notes[-1].end = data.end
        else:
            new_data.append(data)
            new_midi_notes.append(midi_segments[i])
        previous_data = data
    return new_midi_notes, new_data
