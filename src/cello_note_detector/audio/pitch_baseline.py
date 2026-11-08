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
# Longer min duration + wider median window + same-pitch bridging reduce
# vibrato / bow-noise splits that shatter sustained notes into clutter.
DEFAULT_MIN_DURATION = 0.08
DEFAULT_GAP_TOLERANCE = 0.10
DEFAULT_MEDIAN_FILTER_SIZE = 11
DEFAULT_ONSET_WAIT = 0.09
DEFAULT_ONSET_DELTA = 0.11
# Ignore weak spectral peaks (sustained drones / vibrato). Real note
# attacks on the synthetic phrases sit well above this; sample-pack
# drones mostly do not.
DEFAULT_MIN_ONSET_STRENGTH = 1.15
# Merge adjacent ±1-semitone flicker shorter than this into neighbours.
# Keep this below typical rearticulation / neighbor-tone lengths so a
# short F♯ island does not glue two articulated G's into one blob.
# Also keep below ~sixteenth neighbor tones (C–B–C) in continuous streams.
DEFAULT_FLICKER_MERGE_SEC = 0.10
DEFAULT_FLICKER_SEMI_TONES = 1
# Same-MIDI run bridging: absorb short fragments / vibrato splits while
# keeping intentional rearticulations (e.g. Twinkle C–C) when every note
# in the run is long and gaps are articulation-sized.
DEFAULT_SAME_PITCH_MAX_GAP_SEC = 0.14
DEFAULT_SAME_PITCH_SHORT_SEC = 0.28
# Minimum sounding length before an onset may re-split the same pitch.
# Slightly shorter so continuous sixteenth streams can rearticulate.
DEFAULT_ONSET_SPLIT_MIN_NOTE_SEC = 0.12
# Near-duplicate ±1 absorb: only true micro-islands (not neighbor tones).
# Must stay below typical sixteenth lengths (~0.15–0.25 s) so written
# neighbor figures (e.g. C–B–C) are not folded into one pitch.
DEFAULT_NEAR_DUP_GAP_SEC = 0.06
DEFAULT_NEAR_DUP_GUEST_SEC = 0.08


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
    onset_split_min_note_sec: float = DEFAULT_ONSET_SPLIT_MIN_NOTE_SEC,
) -> list[NoteEvent]:
    """Collapse F0 frames into note events with pitch-change + onset splits.

    Rules (monophonic melody):
    - Contiguous same-MIDI voiced frames merge into one event.
    - Brief unvoiced gaps ≤ ``gap_tolerance`` do not end the note if pitch
      resumes at the same MIDI (bow noise / pyin dropouts).
    - MIDI pitch changes start a new event.
    - An onset peak near a frame boundary also starts a new event, so
      rearticulated notes of the same pitch are separated — but only once
      the current note is already long enough (avoids vibrato micro-splits).
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
                # Split on a fresh onset after the note has already sounded
                # long enough (skip the attack frame; ignore early vibrato
                # energy that would shatter a sustain into fragments).
                elapsed = float(times[i]) - float(times[start])
                if (
                    i > start
                    and elapsed >= onset_split_min_note_sec
                    and _nearest_onset(
                        onsets, float(times[i]), max_dist=onset_split_window
                    )
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


def merge_pitch_flicker(
    events: list[NoteEvent],
    *,
    max_flicker_sec: float = DEFAULT_FLICKER_MERGE_SEC,
    max_semitones: int = DEFAULT_FLICKER_SEMI_TONES,
) -> list[NoteEvent]:
    """Merge short ±N-semitone neighbors into the surrounding stable pitch.

    Vibrato / pyin jitter often emits G–F#–G micro-notes; folding those
    into the dominant pitch keeps rhythm quantization from inventing 32nds.

    Long same-pitch rearticulations are *not* glued here — that is handled
    by ``merge_same_pitch_runs`` with an explicit short-fragment rule so
    Twinkle-style repeats stay separate.
    """
    if len(events) < 2 or max_flicker_sec <= 0:
        return events

    ordered = sorted(events, key=lambda e: (e.onset, e.offset))

    # Pass 1: same-pitch short fragments only. ±1 islands are handled in
    # the sandwich pass so a boundary F♯ does not glue two articulated G's.
    merged: list[NoteEvent] = [ordered[0]]
    for ev in ordered[1:]:
        prev = merged[-1]
        gap = float(ev.onset) - float(prev.offset)
        semi = abs(int(ev.midi) - int(prev.midi))
        touch = gap <= max(0.03, max_flicker_sec * 0.5)
        if (
            touch
            and semi == 0
            and min(prev.duration, ev.duration) <= max_flicker_sec * 1.25
        ):
            merged[-1] = _combine_note_events([prev, ev])
        else:
            merged.append(ev)

    # Pass 2: G–(short F#)–G sandwich → one G when the sides look like
    # fragments of one sustain. Two already-long sides with a blip between
    # are treated as articulated notes (keep the island / boundary).
    if len(merged) < 3:
        return merged
    out: list[NoteEvent] = [merged[0]]
    i = 1
    while i < len(merged):
        cur = merged[i]
        nxt = merged[i + 1] if i + 1 < len(merged) else None
        sides_are_long = (
            nxt is not None
            and out[-1].duration > 0.40
            and nxt.duration > 0.40
        )
        if (
            nxt is not None
            and not sides_are_long
            and cur.duration <= max_flicker_sec
            and out[-1].midi == nxt.midi
            and abs(cur.midi - out[-1].midi) <= max_semitones
            and (cur.onset - out[-1].offset) <= max_flicker_sec * 0.75
            and (nxt.onset - cur.offset) <= max_flicker_sec * 0.75
        ):
            host = out[-1]
            onset = min(host.onset, cur.onset, nxt.onset)
            offset = max(host.offset, cur.offset, nxt.offset)
            w0 = max(host.duration, 1e-6)
            w1 = max(nxt.duration, 1e-6)
            conf = (host.confidence * w0 + nxt.confidence * w1) / (w0 + w1)
            out[-1] = NoteEvent(
                pitch_hz=host.pitch_hz,
                midi=host.midi,
                note=host.note,
                onset=round(float(onset), 4),
                offset=round(float(offset), 4),
                duration=round(float(offset - onset), 4),
                confidence=round(float(conf), 4),
            )
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _combine_note_events(parts: list[NoteEvent]) -> NoteEvent:
    """Merge ordered note fragments into one event (host = longest)."""
    host = max(parts, key=lambda e: e.duration)
    onset = min(e.onset for e in parts)
    offset = max(e.offset for e in parts)
    conf_num = 0.0
    conf_den = 0.0
    hz_num = 0.0
    for e in parts:
        w = max(e.duration, 1e-6)
        conf_num += e.confidence * w
        hz_num += e.pitch_hz * w
        conf_den += w
    return NoteEvent(
        pitch_hz=round(float(hz_num / conf_den), 4),
        midi=host.midi,
        note=host.note,
        onset=round(float(onset), 4),
        offset=round(float(offset), 4),
        duration=round(float(offset - onset), 4),
        confidence=round(float(conf_num / conf_den), 4),
    )


def merge_same_pitch_runs(
    events: list[NoteEvent],
    *,
    max_gap_sec: float = DEFAULT_SAME_PITCH_MAX_GAP_SEC,
    short_sec: float = DEFAULT_SAME_PITCH_SHORT_SEC,
) -> list[NoteEvent]:
    """Merge same-MIDI neighbors when a short fragment bridges a small gap.

    Pairwise (iterative): merge only when gap ≤ ``max_gap_sec`` and at
    least one side is ≤ ``short_sec``. Long+long pairs (Twinkle-style
    reattacks, even with a zero gap after onset splitting) stay separate.
    """
    if len(events) < 2 or max_gap_sec <= 0:
        return events

    ordered = sorted(events, key=lambda e: (e.onset, e.offset))
    changed = True
    while changed:
        changed = False
        out: list[NoteEvent] = [ordered[0]]
        for ev in ordered[1:]:
            prev = out[-1]
            gap = float(ev.onset) - float(prev.offset)
            if ev.midi != prev.midi or gap > max_gap_sec:
                out.append(ev)
                continue
            if min(prev.duration, ev.duration) <= short_sec:
                out[-1] = _combine_note_events([prev, ev])
                changed = True
            else:
                out.append(ev)
        ordered = out
    return ordered


def merge_adjacent_near_duplicates(
    events: list[NoteEvent],
    *,
    max_gap_sec: float = DEFAULT_NEAR_DUP_GAP_SEC,
    max_semitones: int = 1,
    max_guest_sec: float = DEFAULT_NEAR_DUP_GUEST_SEC,
) -> list[NoteEvent]:
    """Fold a short ±1-semitone neighbor into a longer stable pitch.

    Complements ``merge_pitch_flicker`` after same-pitch bridging so leftover
    vibrato islands still collapse into readable monophonic notes. Guests
    longer than ``max_guest_sec`` are kept (real neighbor tones / approaches).
    """
    if len(events) < 2:
        return events
    ordered = sorted(events, key=lambda e: (e.onset, e.offset))
    merged: list[NoteEvent] = [ordered[0]]
    for ev in ordered[1:]:
        prev = merged[-1]
        gap = float(ev.onset) - float(prev.offset)
        semi = abs(int(ev.midi) - int(prev.midi))
        if gap > max_gap_sec or semi > max_semitones:
            merged.append(ev)
            continue
        if semi == 0:
            # Only glue same-pitch pairs when at least one side is short /
            # fragmentary — long rearticulations stay separate.
            if min(prev.duration, ev.duration) <= max_guest_sec * 1.5:
                merged[-1] = _combine_note_events([prev, ev])
            else:
                merged.append(ev)
            continue
        # ±1: absorb only true micro-islands into the longer host.
        # Require the guest to be clearly fragmentary AND much shorter than
        # the host so articulated neighbor tones in equal-note streams survive.
        guest_ok = max_guest_sec
        if (
            ev.duration <= guest_ok
            and prev.duration >= max(ev.duration * 2.0, guest_ok * 2.0)
        ):
            merged[-1] = NoteEvent(
                pitch_hz=prev.pitch_hz,
                midi=prev.midi,
                note=prev.note,
                onset=round(float(prev.onset), 4),
                offset=round(float(max(prev.offset, ev.offset)), 4),
                duration=round(
                    float(max(prev.offset, ev.offset) - prev.onset), 4
                ),
                confidence=prev.confidence,
            )
        elif (
            prev.duration <= guest_ok
            and ev.duration >= max(prev.duration * 2.0, guest_ok * 2.0)
        ):
            merged[-1] = NoteEvent(
                pitch_hz=ev.pitch_hz,
                midi=ev.midi,
                note=ev.note,
                onset=round(float(min(prev.onset, ev.onset)), 4),
                offset=round(float(ev.offset), 4),
                duration=round(
                    float(ev.offset - min(prev.onset, ev.onset)), 4
                ),
                confidence=ev.confidence,
            )
        else:
            merged.append(ev)
    return merged


def transcribe_notes(
    audio: np.ndarray,
    *,
    sr: int = TARGET_SR,
    min_duration: float = DEFAULT_MIN_DURATION,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    median_filter_size: int = DEFAULT_MEDIAN_FILTER_SIZE,
    use_onsets: bool = True,
    min_onset_strength: float = DEFAULT_MIN_ONSET_STRENGTH,
    merge_flicker: bool = True,
    flicker_merge_sec: float = DEFAULT_FLICKER_MERGE_SEC,
    merge_same_pitch: bool = True,
    same_pitch_max_gap_sec: float = DEFAULT_SAME_PITCH_MAX_GAP_SEC,
    same_pitch_short_sec: float = DEFAULT_SAME_PITCH_SHORT_SEC,
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
    raw_count = len(events)
    if merge_flicker:
        events = merge_pitch_flicker(
            events, max_flicker_sec=flicker_merge_sec
        )
    if merge_same_pitch:
        events = merge_same_pitch_runs(
            events,
            max_gap_sec=same_pitch_max_gap_sec,
            short_sec=same_pitch_short_sec,
        )
        events = merge_adjacent_near_duplicates(events)
    meta = {
        "frame_length": frame_length,
        "hop_length": hop_length,
        "min_duration": min_duration,
        "gap_tolerance": gap_tolerance,
        "median_filter_size": median_filter_size,
        "use_onsets": use_onsets,
        "min_onset_strength": min_onset_strength if use_onsets else None,
        "merge_flicker": merge_flicker,
        "flicker_merge_sec": flicker_merge_sec if merge_flicker else None,
        "merge_same_pitch": merge_same_pitch,
        "same_pitch_max_gap_sec": (
            same_pitch_max_gap_sec if merge_same_pitch else None
        ),
        "same_pitch_short_sec": (
            same_pitch_short_sec if merge_same_pitch else None
        ),
        "num_notes_pre_merge": raw_count,
        "num_onsets": int(len(onset_times)),
        "onset_times": [round(float(t), 4) for t in onset_times.tolist()],
    }
    return events, meta
