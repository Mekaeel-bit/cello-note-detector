"""Unit tests for key-signature inference and CLI parsing."""

from __future__ import annotations

from cello_note_detector.decode.events import NoteEvent
from cello_note_detector.decode.key import infer_key, parse_key


def _note(midi: int, onset: float, duration: float, conf: float = 0.9) -> NoteEvent:
    return NoteEvent(
        pitch_hz=220.0,
        midi=midi,
        note="X",
        onset=onset,
        offset=onset + duration,
        duration=duration,
        confidence=conf,
    )


def test_parse_key_gm():
    k = parse_key("Gm")
    assert k["name"] == "G minor"
    assert k["fifths"] == -2
    assert k["mode"] == "minor"


def test_parse_key_bb_major():
    k = parse_key("Bb")
    assert k["tonic"] == "Bb"
    assert k["mode"] == "major"
    assert k["fifths"] == -2


def test_infer_g_minor_from_pitches():
    # Duration-weighted G minor triad + flat 7 (F).
    notes = [
        _note(43, 0.0, 1.0),   # G2
        _note(46, 1.0, 0.8),   # Bb2
        _note(50, 1.8, 0.8),   # D3
        _note(41, 2.6, 0.6),   # F2
        _note(43, 3.2, 1.0),   # G2
    ]
    key = infer_key(notes)
    assert key["tonic"] == "G"
    assert key["mode"] == "minor"
    assert key["fifths"] == -2


def test_infer_c_major_default_when_empty():
    key = infer_key([])
    assert key["name"] == "C major"
    assert key["fifths"] == 0
