"""Unit tests for pitch→string hypotheses and sticky AV fusion."""

from __future__ import annotations

from cello_note_detector.video.fusion import fuse_timeline
from cello_note_detector.video.strings import (
    OPEN_STRING_MIDI,
    finger_for_stop,
    pitch_string_hypotheses,
    primary_hypothesis,
)


def test_open_strings_are_primary_open():
    for midi, name in OPEN_STRING_MIDI.items():
        hyp = primary_hypothesis(midi)
        assert hyp is not None
        assert hyp.string == name
        assert hyp.semitones_above_open == 0
        assert hyp.suggested_finger == 0
        assert hyp.position_band == "open"


def test_a3_also_reachable_on_lower_strings():
    hyps = pitch_string_hypotheses(57)
    strings = [h.string for h in hyps]
    assert strings[0] == "A"
    assert "D" in strings and "G" in strings and "C" in strings
    assert hyps[0].semitones_above_open == 0


def test_first_position_finger_map():
    # G2 + 2 semitones → A2 on G string, finger 1
    hyp = primary_hypothesis(45)  # A2
    assert hyp is not None
    assert hyp.string == "G"
    assert hyp.semitones_above_open == 2
    assert hyp.suggested_finger == 1


def test_finger_for_stop_never_open_unless_zero():
    assert finger_for_stop(0, position="I") == 0
    assert finger_for_stop(2, position="I") == 1
    assert finger_for_stop(4, position="I") == 2
    # B3 on A (=2 st) must not be open
    assert finger_for_stop(2, position="I") != 0
    # High stop unreachable in I
    assert finger_for_stop(12, position="I") is None
    assert finger_for_stop(12, position="thumb") in (1, 2, 3, 4)


def test_fuse_timeline_labels_bow_string():
    timeline = {
        "schema": "cello_video_timeline_v0",
        "source": "dummy.mp4",
        "sample_every": 1,
        "limits": [],
        "events": [
            {"t": 0.5, "type": "bow", "direction": "down", "position": "middle"},
            {"t": 0.5, "type": "finger_on", "finger": 1},
        ],
        "samples": [
            {
                "t": 0.5,
                "frame_idx": 15,
                "active_fingers": [1],
                "bow": {"direction": "down", "position": "middle", "vx": -0.2, "vy": 0.0},
                "left_hand": {
                    "wrist": {"x": 0.4, "y": 0.22},
                    "tips": {
                        "index": {"x": 0.42, "y": 0.20},
                        "middle": {"x": 0.43, "y": 0.21},
                        "ring": {"x": 0.44, "y": 0.22},
                        "pinky": {"x": 0.45, "y": 0.23},
                    },
                },
                "right_hand": {"wrist": {"x": 0.6, "y": 0.7}},
            }
        ],
        "audio_notes": [
            {
                "pitch_hz": 220.0,
                "midi": 57,
                "note": "A3",
                "onset": 0.0,
                "offset": 1.0,
                "duration": 1.0,
                "confidence": 0.9,
            }
        ],
    }
    fused = fuse_timeline(timeline)
    assert fused["schema"] == "cello_av_fusion_v1"
    assert fused["fusion"]["ok"] is True
    assert fused["fused_notes"][0]["string"] == "A"
    assert fused["fused_notes"][0]["fingering"]["finger"] == 0
    assert fused["fused_samples"][0]["bow"]["string"] == "A"


def test_fuse_a_string_scale_does_not_collapse_to_open():
    """Climbing pitches on A must get finger 1–4, not open-A after A3."""
    # Sparse samples only near the start (simulates --max-frames truncating vision).
    samples = [
        {
            "t": t,
            "frame_idx": i,
            "active_fingers": [],
            "bow": {"direction": "down", "position": "middle", "vx": -0.1, "vy": 0.0},
            "left_hand": {
                "wrist": {"x": 0.4, "y": 0.20 + 0.01 * i},
                "tips": {
                    "index": {"x": 0.42, "y": 0.18 + 0.01 * i},
                    "middle": {"x": 0.43, "y": 0.19 + 0.01 * i},
                    "ring": {"x": 0.44, "y": 0.20 + 0.01 * i},
                    "pinky": {"x": 0.45, "y": 0.21 + 0.01 * i},
                },
            },
            "right_hand": {"wrist": {"x": 0.6, "y": 0.7}},
        }
        for i, t in enumerate([0.0, 0.5, 1.0, 1.5])
    ]
    # A3 open, then B C# D E F# on A string (and beyond vision window).
    midis = [57, 59, 61, 62, 64, 66, 68, 69]
    notes = []
    for i, midi in enumerate(midis):
        onset = float(i)
        notes.append(
            {
                "pitch_hz": 440.0 * (2 ** ((midi - 69) / 12)),
                "midi": midi,
                "note": f"m{midi}",
                "onset": onset,
                "offset": onset + 0.8,
                "duration": 0.8,
                "confidence": 0.9,
            }
        )
    fused = fuse_timeline(
        {
            "schema": "cello_video_timeline_v0",
            "source": "dummy.mp4",
            "limits": [],
            "events": [],
            "samples": samples,
            "audio_notes": notes,
        }
    )
    rows = fused["fused_notes"]
    assert rows[0]["string"] == "A" and rows[0]["fingering"]["finger"] == 0
    assert rows[0]["lh_position"] == "I"
    # Next notes stay on A with non-zero fingers (not open).
    for row in rows[1:5]:
        assert row["string"] == "A", row
        assert row["fingering"]["finger"] != 0, row
        assert row["semitones_above_open"] == row["midi"] - 57
    # Higher notes may shift position but must not all become finger 0.
    assert sum(1 for r in rows if r["fingering"]["finger"] == 0) == 1
