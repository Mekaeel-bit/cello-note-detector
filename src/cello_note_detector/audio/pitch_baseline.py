"""Classical F0 baseline via librosa.pyin (monophonic cello range)."""

from __future__ import annotations

import librosa
import numpy as np

from cello_note_detector.decode.events import NoteEvent

from .load import TARGET_SR

# Approximate cello playing range for pyin bounds.
CELLO_FMIN = "C2"
CELLO_FMAX = "A5"
DEFAULT_FRAME_LENGTH = 4096
DEFAULT_HOP_LENGTH = 512


def detect_f0(
    audio: np.ndarray,
    *,
    sr: int = TARGET_SR,
    fmin: str = CELLO_FMIN,
    fmax: str = CELLO_FMAX,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run pyin and return ``(times, f0, voiced_flag, voiced_probs)``."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    f0 = np.asarray(f0, dtype=np.float64)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_probs = np.asarray(voiced_probs, dtype=np.float64)
    return times, f0, voiced_flag, voiced_probs


def f0_to_note_events(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    sr: int = TARGET_SR,
    min_duration: float = 0.05,
) -> list[NoteEvent]:
    """Collapse contiguous same-MIDI voiced frames into note events.

    Frames with the same rounded MIDI pitch are merged. Gaps or pitch
    changes start a new event. Events shorter than ``min_duration`` are
    dropped (helps suppress pyin flicker).
    """
    frame_dur = hop_length / float(sr)
    events: list[NoteEvent] = []

    i = 0
    n = len(f0)
    while i < n:
        if not (voiced_flag[i] and np.isfinite(f0[i]) and f0[i] > 0):
            i += 1
            continue

        midi = int(round(librosa.hz_to_midi(float(f0[i]))))
        start = i
        hz_vals: list[float] = []
        conf_vals: list[float] = []

        while i < n:
            ok = voiced_flag[i] and np.isfinite(f0[i]) and f0[i] > 0
            if not ok:
                break
            frame_midi = int(round(librosa.hz_to_midi(float(f0[i]))))
            if frame_midi != midi:
                break
            hz_vals.append(float(f0[i]))
            conf_vals.append(float(voiced_probs[i]))
            i += 1

        onset = float(times[start])
        offset = float(times[i - 1]) + frame_dur
        duration = offset - onset
        if duration < min_duration:
            continue

        mean_hz = float(np.mean(hz_vals))
        events.append(
            NoteEvent(
                pitch_hz=round(mean_hz, 4),
                midi=midi,
                note=librosa.midi_to_note(midi),
                onset=round(onset, 4),
                offset=round(offset, 4),
                duration=round(duration, 4),
                confidence=round(float(np.mean(conf_vals)), 4),
            )
        )

    return events
