"""Unit tests for pitch→string hypotheses and AV fusion helpers."""

from __future__ import annotations

from cello_note_detector.video.fusion import fuse_timeline
from cello_note_detector.video.strings import (
    OPEN_STRING_MIDI,
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
    # A3 (57) is open A, but also high on D/G/C.
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
                    "wrist": {"x": 0.4, "y": 0.28},
                    "tips": {
                        "index": {"x": 0.42, "y": 0.25},
                        "middle": {"x": 0.43, "y": 0.26},
                        "ring": {"x": 0.44, "y": 0.27},
                        "pinky": {"x": 0.45, "y": 0.28},
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
    assert fused["schema"] == "cello_av_fusion_v0"
    assert fused["fusion"]["ok"] is True
    assert fused["fused_notes"]
    assert fused["fused_notes"][0]["string"] == "A"
    assert fused["fused_samples"][0]["bow"]["string"] == "A"
    assert fused["events"][0].get("string") == "A" or fused["events"][0].get("sounding_string") == "A"
    fing = fused["fused_samples"][0]["fingering"]
    assert fing is not None
    assert fing["finger"] == 0  # open A
