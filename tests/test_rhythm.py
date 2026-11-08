"""Unit tests for tempo quantization and rest extraction."""

from __future__ import annotations

import numpy as np

from cello_note_detector.decode.events import NoteEvent
from cello_note_detector.decode.rhythm import (
    analyze_rhythm,
    extract_rests,
    nearest_duration_value,
    quantize_interval,
    seconds_to_beats,
)


def _note(midi: int, onset: float, duration: float) -> NoteEvent:
    return NoteEvent(
        pitch_hz=440.0,
        midi=midi,
        note="A4",
        onset=onset,
        offset=onset + duration,
        duration=duration,
        confidence=0.9,
    )


def test_nearest_duration_quarter():
    name, beats = nearest_duration_value(1.02)
    assert name == "quarter"
    assert beats == 1.0


def test_quantize_interval_matches_name_and_beats():
    q = quantize_interval(0.0, 0.6, bpm=100.0)  # 1.0 beat
    assert q.duration_value == "quarter"
    assert q.quantized_duration_beats == 1.0
    assert q.duration_value_beats == 1.0


def test_extract_rests_inter_note_and_edges():
    notes = [_note(48, 0.5, 0.5), _note(52, 1.6, 0.5)]  # 0.6s gap
    rests = extract_rests(notes, audio_duration_sec=2.5, min_rest_sec=0.08)
    assert rests[0].onset == 0.0 and rests[0].offset == 0.5
    mid = [r for r in rests if abs(r.duration - 0.6) < 1e-6]
    assert len(mid) == 1
    assert rests[-1].offset == 2.5


def test_analyze_rhythm_quarters_with_rests():
    # Synthetic: quarter, quarter-rest, quarter, quarter-rest, half @ 100 BPM.
    sr = 24_000
    bpm = 100.0
    beat = 60.0 / bpm
    notes = [
        _note(48, 0.05, beat),
        _note(52, 0.05 + 2 * beat, beat),
        _note(55, 0.05 + 4 * beat, 2 * beat),
    ]
    # Quiet noise floor so librosa still runs.
    audio = np.zeros(int(7 * beat * sr), dtype=np.float32)
    # Put short tone bursts so beat_track has energy (optional).
    for n in notes:
        a = int(n.onset * sr)
        b = int(n.offset * sr)
        t = np.arange(max(0, b - a), dtype=np.float32) / sr
        audio[a:b] = 0.2 * np.sin(2 * np.pi * 130.8 * t)

    result = analyze_rhythm(audio, notes, sr=sr, min_rest_sec=0.08)
    assert result["num_rests"] >= 2
    # Inter-note rests should quantize near a quarter.
    inter = [
        r
        for r in result["rests"]
        if r["onset"] > 0.05 and r["offset"] < len(audio) / sr - 0.05
    ]
    assert len(inter) == 2
    for r in inter:
        assert r["duration_value"] == "quarter"
        assert r["quantized_duration_beats"] == 1.0

    note_vals = [nr["duration_value"] for nr in result["note_rhythm"]]
    assert note_vals[0] == "quarter"
    assert note_vals[1] == "quarter"
    assert note_vals[2] == "half"
    assert abs(result["tempo"]["bpm"] - 100.0) < 15.0


def test_seconds_to_beats():
    assert abs(seconds_to_beats(0.6, 100.0) - 1.0) < 1e-9


def test_coarse_catalog_rejects_thirty_seconds():
    from cello_note_detector.decode.rhythm import (
        DURATION_CATALOG_COARSE,
        nearest_duration_value,
        resolve_quantize_preset,
    )

    name, beats = nearest_duration_value(
        0.14, catalog=DURATION_CATALOG_COARSE, min_beats=0.5
    )
    assert name == "eighth"
    assert beats == 0.5
    cfg = resolve_quantize_preset("coarse")
    assert cfg["grid_beats"] == 0.5
    assert "thirty_second" not in {n for _b, n in cfg["catalog"]}


def test_bpm_override_skips_estimation():
    sr = 24_000
    audio = np.zeros(sr, dtype=np.float32)
    notes = [_note(48, 0.1, 0.4), _note(50, 0.6, 0.4)]
    result = analyze_rhythm(
        audio, notes, sr=sr, bpm_override=120.0, min_rest_sec=0.08
    )
    assert abs(result["tempo"]["bpm"] - 120.0) < 0.01
    assert result["tempo"]["method"] == "cli_override"
