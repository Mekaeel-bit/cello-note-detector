"""Unit tests for pitch flicker / same-pitch merge helpers."""

from __future__ import annotations

from cello_note_detector.audio.pitch_baseline import (
    merge_adjacent_near_duplicates,
    merge_pitch_flicker,
    merge_same_pitch_runs,
)
from cello_note_detector.decode.events import NoteEvent
import librosa


def _ev(midi: int, onset: float, duration: float, conf: float = 0.9) -> NoteEvent:
    return NoteEvent(
        pitch_hz=round(float(librosa.midi_to_hz(midi)), 4),
        midi=midi,
        note=librosa.midi_to_note(midi),
        onset=onset,
        offset=onset + duration,
        duration=duration,
        confidence=conf,
    )


def test_same_pitch_merges_short_fragment_only():
    # Long–short pair merges; a following long note stays unless it is also short.
    notes = [
        _ev(38, 0.0, 0.40),
        _ev(38, 0.42, 0.18),
        _ev(38, 0.70, 0.90),
    ]
    merged = merge_same_pitch_runs(notes, max_gap_sec=0.14, short_sec=0.30)
    assert len(merged) == 2
    assert merged[0].midi == 38
    assert abs(merged[0].onset - 0.0) < 1e-6
    assert abs(merged[0].offset - 0.60) < 1e-6
    assert abs(merged[1].duration - 0.90) < 1e-6


def test_same_pitch_keeps_twinkle_style_reattacks():
    # Two long equal notes with a tiny gap must stay separate.
    notes = [_ev(48, 0.0, 0.45), _ev(48, 0.45, 0.45)]
    merged = merge_same_pitch_runs(notes, max_gap_sec=0.14, short_sec=0.30)
    assert len(merged) == 2


def test_flicker_sandwich_skips_two_long_sides():
    # Articulated G's with a short F# between should not become one blob.
    notes = [
        _ev(43, 0.0, 0.55),
        _ev(42, 0.55, 0.10),
        _ev(43, 0.68, 0.55),
    ]
    merged = merge_pitch_flicker(notes, max_flicker_sec=0.12)
    assert len(merged) == 3


def test_flicker_sandwich_merges_short_sides():
    notes = [
        _ev(43, 0.0, 0.20),
        _ev(42, 0.20, 0.08),
        _ev(43, 0.30, 0.25),
    ]
    merged = merge_pitch_flicker(notes, max_flicker_sec=0.12)
    assert len(merged) == 1
    assert merged[0].midi == 43


def test_near_dup_absorbs_micro_island_not_neighbor_tone():
    # Short F# after G folds in; Bb–A neighbor tones of similar length stay.
    g_fs = [_ev(43, 0.0, 0.60), _ev(42, 0.62, 0.12)]
    folded = merge_adjacent_near_duplicates(g_fs, max_guest_sec=0.20)
    assert len(folded) == 1 and folded[0].midi == 43

    bb_a = [_ev(46, 0.0, 0.40), _ev(45, 0.42, 0.40)]
    kept = merge_adjacent_near_duplicates(bb_a, max_guest_sec=0.20)
    assert len(kept) == 2
