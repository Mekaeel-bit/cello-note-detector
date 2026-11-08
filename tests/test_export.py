"""Unit tests for MusicXML / MIDI sheet export."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

from cello_note_detector.export.midi import write_midi
from cello_note_detector.export.musicxml import (
    build_musicxml,
    midi_to_pitch,
    write_musicxml,
)


def _sample_result() -> dict:
    """Quarter, quarter-rest, quarter, quarter-rest, half @ 100 BPM."""
    return {
        "audio_path": "data/multi_note_phrases/rhythm_quarters_with_rests.wav",
        "tempo": {"bpm": 100.0},
        "notes": [
            {
                "midi": 48,
                "note": "C3",
                "onset": 0.05,
                "offset": 0.65,
                "duration": 0.6,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "quantized_duration_beats": 1.0,
            },
            {
                "midi": 52,
                "note": "E3",
                "onset": 1.25,
                "offset": 1.85,
                "duration": 0.6,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "quantized_duration_beats": 1.0,
            },
            {
                "midi": 55,
                "note": "G3",
                "onset": 2.45,
                "offset": 3.65,
                "duration": 1.2,
                "duration_value": "half",
                "duration_value_beats": 2.0,
                "quantized_duration_beats": 2.0,
            },
        ],
        "rests": [
            {
                "kind": "rest",
                "onset": 0.65,
                "offset": 1.25,
                "duration": 0.6,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "quantized_duration_beats": 1.0,
            },
            {
                "kind": "rest",
                "onset": 1.85,
                "offset": 2.45,
                "duration": 0.6,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "quantized_duration_beats": 1.0,
            },
        ],
        "events": [
            {
                "type": "note",
                "midi": 48,
                "note": "C3",
                "onset": 0.05,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "quantized_duration_beats": 1.0,
            },
            {
                "type": "rest",
                "onset": 0.65,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "quantized_duration_beats": 1.0,
            },
            {
                "type": "note",
                "midi": 52,
                "note": "E3",
                "onset": 1.25,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "quantized_duration_beats": 1.0,
            },
            {
                "type": "rest",
                "onset": 1.85,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "quantized_duration_beats": 1.0,
            },
            {
                "type": "note",
                "midi": 55,
                "note": "G3",
                "onset": 2.45,
                "duration_value": "half",
                "duration_value_beats": 2.0,
                "quantized_duration_beats": 2.0,
            },
        ],
    }


def test_midi_to_pitch_cello_range():
    assert midi_to_pitch(36) == ("C", 0, 2)  # C2
    assert midi_to_pitch(48) == ("C", 0, 3)
    assert midi_to_pitch(50) == ("D", 0, 3)
    assert midi_to_pitch(51) == ("D", 1, 3)  # D#
    assert midi_to_pitch(69) == ("A", 0, 4)


def test_musicxml_bass_clef_and_rests(tmp_path: Path):
    result = _sample_result()
    root = build_musicxml(result)
    assert root.tag == "score-partwise"

    clef_sign = root.find(".//clef/sign")
    assert clef_sign is not None and clef_sign.text == "F"
    clef_line = root.find(".//clef/line")
    assert clef_line is not None and clef_line.text == "4"

    pitches = [
        (p.find("step").text, p.find("octave").text)
        for p in root.findall(".//pitch")
    ]
    assert pitches == [("C", "3"), ("E", "3"), ("G", "3")]

    rests = root.findall(".//rest")
    # 2 intentional rests + possible final-measure pad rests
    assert len(rests) >= 2

    types = [n.find("type").text for n in root.findall(".//note") if n.find("type") is not None]
    assert "quarter" in types
    assert "half" in types

    out = write_musicxml(result, tmp_path / "phrase.musicxml")
    assert out.exists()
    parsed = ET.parse(out)
    assert parsed.getroot().tag == "score-partwise"


def test_midi_file_header_and_notes(tmp_path: Path):
    result = _sample_result()
    path = write_midi(result, tmp_path / "phrase.mid")
    data = path.read_bytes()
    assert data[:4] == b"MThd"
    # header length 6, format 1, 1 track
    length, fmt, ntrks, ppq = struct.unpack(">IHHH", data[4:14])
    assert length == 6
    assert fmt == 1
    assert ntrks == 1
    assert ppq == 480
    assert b"MTrk" in data
    # Note-on for C3 (48), E3 (52), G3 (55) should appear.
    assert bytes([0x90, 48]) in data or bytes([0x90, 48, 80]) in data
    assert bytes([0x90, 52, 80]) in data
    assert bytes([0x90, 55, 80]) in data
