"""Classical F0 baseline via librosa.pyin (monophonic cello range)."""

from __future__ import annotations

import librosa
import numpy as np

from cello_note_detector.decode.events import NoteEvent

from .load import TARGET_SR

# Approximate cello playing range for pyin bounds.
CELLO_FMIN = "C2"
CELLO_FMAX = "A5"
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 256

# Segmentation defaults tuned for short monophonic cello phrases.
DEFAULT_MIN_DURATION = 0.04
DEFAULT_GAP_TOLERANCE = 0.06
DEFAULT_MEDIAN_FILTER_SIZE = 5
DEFAULT_ONSET_WAIT = 0.08
DEFAULT_ONSET_DELTA = 0.1
# Ignore weak spectral peaks (sustained drones / vibrato). Real note
# attacks on the synthetic phrases sit well above this; sample-pack
# drones mostly do not.
DEFAULT_MIN_ONSET_STRENGTH = 1.2


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


def detect_onsets(
    audio: np.ndarray,
    *,
    sr: int = TARGET_SR,
    hop_length: int = DEFAULT_HOP_LENGTH,
    wait: float = DEFAULT_ONSET_WAIT,
    delta: float = DEFAULT_ONSET_DELTA,
    min_strength: float = DEFAULT_MIN_ONSET_STRENGTH,
) -> np.ndarray:
    """Return onset times (seconds) via spectral-flux onset strength.

    Peaks below ``min_strength`` are dropped so vibrato / bow noise on a
    sustained pitch does not shatter one note into many events.
    """
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=hop_length
    )
    wait_frames = max(1, int(round(wait * sr / hop_length)))
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
        wait=wait_frames,
        delta=delta,
        pre_max=wait_frames,
        post_max=wait_frames,
        pre_avg=wait_frames * 2,
        post_avg=wait_frames * 2,
    )
    frames = np.asarray(frames, dtype=np.int64)
    if frames.size and min_strength > 0:
        frames = frames[onset_env[frames] >= min_strength]
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)


def _smooth_midi(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    *,
    median_filter_size: int,
) -> np.ndarray:
    """Per-frame MIDI (NaN when unvoiced), optionally median-smoothed."""
    midi = np.full(len(f0), np.nan, dtype=np.float64)
    voiced = voiced_flag & np.isfinite(f0) & (f0 > 0)
    midi[voiced] = np.round(librosa.hz_to_midi(f0[voiced]))

    if median_filter_size <= 1 or not np.any(voiced):
        return midi

    # Median-filter only over voiced frames to kill single-frame MIDI flicker,
    # then scatter back onto the full timeline.
    voiced_idx = np.flatnonzero(voiced)
    voiced_midi = midi[voiced_idx].copy()
    k = int(median_filter_size)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(voiced_midi, (pad, pad), mode="edge")
    smoothed = np.empty_like(voiced_midi)
    for j in range(len(voiced_midi)):
        smoothed[j] = np.median(padded[j : j + k])
    out = midi.copy()
    out[voiced_idx] = np.round(smoothed)
    return out


def _nearest_onset(onset_times: np.ndarray, t: float, *, max_dist: float) -> bool:
    if onset_times.size == 0:
        return False
    return bool(np.min(np.abs(onset_times - t)) <= max_dist)


def f0_to_note_events(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    sr: int = TARGET_SR,
    min_duration: float = DEFAULT_MIN_DURATION,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    median_filter_size: int = DEFAULT_MEDIAN_FILTER_SIZE,
    onset_times: np.ndarray | None = None,
    onset_split_window: float = 0.045,
) -> list[NoteEvent]:
    """Collapse F0 frames into note events with pitch-change + onset splits.

    Rules (monophonic melody):
    - Contiguous same-MIDI voiced frames merge into one event.
    - Brief unvoiced gaps ≤ ``gap_tolerance`` do not end the note if pitch
      resumes at the same MIDI (bow noise / pyin dropouts).
    - MIDI pitch changes start a new event.
    - An onset peak near a frame boundary also starts a new event, so
      rearticulated notes of the same pitch are separated.
    - Events shorter than ``min_duration`` are dropped.
    """
    frame_dur = hop_length / float(sr)
    midi_frames = _smooth_midi(
        f0, voiced_flag, median_filter_size=median_filter_size
    )
    onsets = (
        np.asarray(onset_times, dtype=np.float64)
        if onset_times is not None
        else np.array([], dtype=np.float64)
    )

    events: list[NoteEvent] = []
    i = 0
    n = len(f0)

    while i < n:
        if not np.isfinite(midi_frames[i]):
            i += 1
            continue

        midi = int(midi_frames[i])
        start = i
        hz_vals: list[float] = []
        conf_vals: list[float] = []
        gap_budget = 0.0

        while i < n:
            frame_midi = midi_frames[i]
            voiced_here = np.isfinite(frame_midi)

            if voiced_here:
                if int(frame_midi) != midi:
                    break
                # Split on a fresh onset after the note has already started
                # (skip the first frame so the note's own attack does not
                # immediately re-split).
                if i > start and _nearest_onset(
                    onsets, float(times[i]), max_dist=onset_split_window
                ):
                    break
                hz_vals.append(float(f0[i]))
                conf_vals.append(float(voiced_probs[i]))
                gap_budget = 0.0
                i += 1
                continue

            # Unvoiced: allow a short gap, else end the note.
            gap_budget += frame_dur
            if gap_budget > gap_tolerance:
                break
            i += 1

        if not hz_vals:
            i = max(i, start + 1)
            continue

        # Trim trailing unvoiced frames that were consumed by gap tolerance.
        end_idx = start + len(hz_vals) - 1
        # Find last voiced frame belonging to this note.
        j = i - 1
        while j >= start and not np.isfinite(midi_frames[j]):
            j -= 1
        end_idx = max(start, j)

        onset = float(times[start])
        offset = float(times[end_idx]) + frame_dur
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


def transcribe_notes(
    audio: np.ndarray,
    *,
    sr: int = TARGET_SR,
    min_duration: float = DEFAULT_MIN_DURATION,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    median_filter_size: int = DEFAULT_MEDIAN_FILTER_SIZE,
    use_onsets: bool = True,
    min_onset_strength: float = DEFAULT_MIN_ONSET_STRENGTH,
    frame_length: int = DEFAULT_FRAME_LENGTH,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> tuple[list[NoteEvent], dict]:
    """Full pyin → note-event pipeline with optional onset splitting.

    Returns ``(events, meta)`` where ``meta`` records segmentation settings
    and detected onset times for JSON / debugging.
    """
    times, f0, voiced_flag, voiced_probs = detect_f0(
        audio,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    onset_times = (
        detect_onsets(
            audio,
            sr=sr,
            hop_length=hop_length,
            min_strength=min_onset_strength,
        )
        if use_onsets
        else np.array([], dtype=np.float64)
    )
    events = f0_to_note_events(
        times,
        f0,
        voiced_flag,
        voiced_probs,
        hop_length=hop_length,
        sr=sr,
        min_duration=min_duration,
        gap_tolerance=gap_tolerance,
        median_filter_size=median_filter_size,
        onset_times=onset_times,
    )
    meta = {
        "frame_length": frame_length,
        "hop_length": hop_length,
        "min_duration": min_duration,
        "gap_tolerance": gap_tolerance,
        "median_filter_size": median_filter_size,
        "use_onsets": use_onsets,
        "min_onset_strength": min_onset_strength if use_onsets else None,
        "num_onsets": int(len(onset_times)),
        "onset_times": [round(float(t), 4) for t in onset_times.tolist()],
    }
    return events, meta
