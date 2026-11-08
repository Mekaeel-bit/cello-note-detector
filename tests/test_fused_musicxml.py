"""Tests for fused fingering/position MusicXML export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from cello_note_detector.video.position_map import position_label, annotate_fused_note
from cello_note_detector.video.sheet import build_fused_score_result, write_fused_musicxml
from cello_note_detector.export.musicxml import build_musicxml


def test_position_label_mapping():
    assert position_label(semitones_above_open=0) == "I"
    assert position_label(semitones_above_open=3) == "I"
    assert position_label(semitones_above_open=6) == "II"
    assert position_label(semitones_above_open=9) == "III"
    assert position_label(semitones_above_open=12) == "IV"
    assert position_label(semitones_above_open=16, string_name="A") == "thumb"
    assert position_label(semitones_above_open=16, string_name="C") == "high"


def test_annotate_sets_string_number():
    n = annotate_fused_note(
        {
            "midi": 57,
            "string": "A",
            "semitones_above_open": 0,
            "position_band": "open",
            "fingering": {"finger": 0},
        }
    )
    assert n["score_finger"] == 0
    assert n["score_position"] == "I"
    assert n["score_string_number"] == 1  # A = highest


def test_build_musicxml_emits_fingering_and_position():
    result = {
        "tempo": {"bpm": 100.0},
        "key": {"fifths": 3, "name": "A major"},
        "events": [
            {
                "type": "note",
                "midi": 57,
                "onset": 0.0,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "score_finger": 0,
                "score_string_number": 1,
                "score_position_mark": "I",
            },
            {
                "type": "note",
                "midi": 59,
                "onset": 0.5,
                "duration_value": "quarter",
                "duration_value_beats": 1.0,
                "score_finger": 1,
                "score_string_number": 1,
                "score_position_mark": None,
            },
        ],
    }
    root = build_musicxml(result, title="test")
    xml = ET.tostring(root, encoding="unicode")
    assert "<fingering>0</fingering>" in xml
    assert "<fingering>1</fingering>" in xml
    assert "<string>1</string>" in xml
    assert ">I</words>" in xml or ">I<" in xml


def test_write_fused_musicxml_file(tmp_path: Path):
    fused = {
        "schema": "cello_av_fusion_v0",
        "source": "dummy.mp4",
        "audio_path": None,
        "fused_notes": [
            {
                "midi": 57,
                "note": "A3",
                "pitch_hz": 220.0,
                "onset": 0.0,
                "offset": 0.4,
                "duration": 0.4,
                "confidence": 0.8,
                "string": "A",
                "position_band": "open",
                "semitones_above_open": 0,
                "fingering": {"finger": 0, "label": "open", "confidence": 0.7},
            },
            {
                "midi": 59,
                "note": "B3",
                "pitch_hz": 247.0,
                "onset": 0.45,
                "offset": 0.85,
                "duration": 0.4,
                "confidence": 0.8,
                "string": "A",
                "position_band": "low",
                "semitones_above_open": 2,
                "fingering": {"finger": 1, "label": "finger_1", "confidence": 0.65},
            },
            {
                "midi": 61,
                "note": "C#4",
                "pitch_hz": 277.0,
                "onset": 0.9,
                "offset": 1.3,
                "duration": 0.4,
                "confidence": 0.8,
                "string": "A",
                "position_band": "low",
                "semitones_above_open": 4,
                "fingering": {"finger": 2, "label": "finger_2", "confidence": 0.65},
            },
        ],
    }
    out = tmp_path / "demo_fused.musicxml"
    path, score = write_fused_musicxml(fused, out, title="demo")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "<fingering>" in text
    assert "score-partwise" in text
    assert score["num_notes"] == 3
    # Position mark only on first note of same position (I throughout here).
    marks = [n.get("score_position_mark") for n in score["notes"]]
    assert marks[0] == "I"
    assert marks[1] is None and marks[2] is None
