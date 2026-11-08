"""Tests for score-intent equal-stream rhythm reconstruction."""

from __future__ import annotations

import numpy as np

from cello_note_detector.decode.events import NoteEvent
from cello_note_detector.decode.rhythm import (
    analyze_rhythm,
    detect_equal_note_stream,
    resolve_quantize_preset,
)


def _note(midi: int, onset: float, duration: float) -> NoteEvent:
    return NoteEvent(
        pitch_hz=100.0,
        midi=midi,
        note="X",
        onset=onset,
        offset=onset + duration,
        duration=duration,
        confidence=0.9,
    )


def test_score_preset_enables_score_intent():
    cfg = resolve_quantize_preset("score")
    assert cfg["score_intent"] is True
    assert cfg["grid_beats"] == 0.25


def test_detect_equal_sixteenth_stream():
    # 16 equal onsets @ 0.2 s IOI (sixteenth @ 75 BPM quarter).
    unit = 0.20
    notes = [_note(43 + (i % 5), i * unit, unit * 0.9) for i in range(16)]
    stream = detect_equal_note_stream(notes)
    assert stream is not None
    assert stream["unit_beats"] == 0.25
    assert abs(stream["unit_sec"] - unit) < 0.02


def test_score_intent_maps_held_notes_to_equal_units():
    # Performance flavour: every 4th note held 2×, but score intent is equal 16ths.
    unit = 0.20
    notes = []
    t = 0.0
    for i in range(16):
        dur = unit * 1.8 if i % 4 == 0 else unit * 0.85
        notes.append(_note(43 + (i % 8), t, dur))
        t += unit
    sr = 16_000
    audio = np.zeros(int((t + 0.5) * sr), dtype=np.float32)
    result = analyze_rhythm(
        audio,
        notes,
        sr=sr,
        score_intent=True,
        catalog=resolve_quantize_preset("score")["catalog"],
        grid_beats=0.25,
        min_rest_sec=0.12,
        min_rest_beats=0.25,
        min_note_beats=0.25,
        include_leading_rests=False,
        include_trailing_rests=False,
    )
    assert result["equal_stream"] is not None
    vals = [nr["duration_value"] for nr in result["note_rhythm"]]
    # Most notes should snap to sixteenth despite holds.
    assert vals.count("sixteenth") >= 12
