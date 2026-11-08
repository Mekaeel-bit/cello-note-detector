"""Tempo estimation, beat grid, duration quantization, and rest gaps.

Classical DSP path (librosa beat tracking + spectral/note onsets). Emits
quantized rhythmic metadata for sounding notes and for silences between
them (rests) — sheet music needs both.

Defaults favour readable scores (quarters / eighths) over fine 16th/32nd
grids. Tempo and key are never read from filenames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import librosa
import numpy as np

from cello_note_detector.decode.events import NoteEvent

# Full duration catalog (coarsest → finest). Presets select a prefix/subset.
DURATION_CATALOG_FINE: tuple[tuple[float, str], ...] = (
    (4.0, "whole"),
    (3.0, "dotted_half"),
    (2.0, "half"),
    (1.5, "dotted_quarter"),
    (1.0, "quarter"),
    (0.75, "dotted_eighth"),
    (0.5, "eighth"),
    (0.375, "dotted_sixteenth"),
    (0.25, "sixteenth"),
    (0.125, "thirty_second"),
)

DURATION_CATALOG_MEDIUM: tuple[tuple[float, str], ...] = (
    (4.0, "whole"),
    (3.0, "dotted_half"),
    (2.0, "half"),
    (1.5, "dotted_quarter"),
    (1.0, "quarter"),
    (0.75, "dotted_eighth"),
    (0.5, "eighth"),
    (0.25, "sixteenth"),
)

DURATION_CATALOG_COARSE: tuple[tuple[float, str], ...] = (
    (4.0, "whole"),
    (3.0, "dotted_half"),
    (2.0, "half"),
    (1.5, "dotted_quarter"),
    (1.0, "quarter"),
    (0.75, "dotted_eighth"),
    (0.5, "eighth"),
)

# Score-oriented: favour written equal subdivisions (esp. sixteenths).
DURATION_CATALOG_SCORE: tuple[tuple[float, str], ...] = (
    (4.0, "whole"),
    (3.0, "dotted_half"),
    (2.0, "half"),
    (1.5, "dotted_quarter"),
    (1.0, "quarter"),
    (0.75, "dotted_eighth"),
    (0.5, "eighth"),
    (0.25, "sixteenth"),
)

# Back-compat alias (fine catalog).
DURATION_CATALOG = DURATION_CATALOG_FINE

# Default snap grid: eighth note (was sixteenth). Readable over fussy.
DEFAULT_GRID_BEATS = 0.5
DEFAULT_MIN_BPM = 50.0
DEFAULT_MAX_BPM = 180.0
DEFAULT_MIN_REST_SEC = 0.16
DEFAULT_MIN_REST_BEATS = 0.5
DEFAULT_MIN_NOTE_BEATS = 0.5
# Prefer note-onset / spectral IOI tempo when beat_track disagrees.
IOI_TEMPO_WEIGHT = 0.7
# Ignore sub-articulation IOIs when estimating tempo from onsets.
MIN_TEMPO_IOI_SEC = 0.18
MAX_TEMPO_IOI_SEC = 1.8
# Ignore vibrato flicker when scoring duration-vs-BPM fit.
MIN_TEMPO_NOTE_DUR_SEC = 0.15

# Equal-stream (score-intent) reconstruction: detect moto-perpetuo /
# continuous equal-note writing and snap to the written unit rather than
# raw performance timing (e.g. held open-string flavour → equal sixteenths).
EQUAL_STREAM_MIN_NOTES = 8
EQUAL_STREAM_MAX_IOI_CV = 0.35
EQUAL_STREAM_MIN_IOI_SEC = 0.05
EQUAL_STREAM_MAX_IOI_SEC = 0.90
# Held notes shorter than this × stream unit still map to one written unit
# (performance lengthening / rubato → equal sixteenths). Higher = steadier.
EQUAL_STREAM_HOLD_MULT = 2.5
EQUAL_STREAM_REST_SUPPRESS_MULT = 1.35  # suppress micro-gaps under this × unit

QUANTIZE_PRESETS: dict[str, dict] = {
    "coarse": {
        "grid_beats": 0.5,
        "catalog": DURATION_CATALOG_COARSE,
        "min_rest_sec": 0.18,
        "min_rest_beats": 0.5,
        "min_note_beats": 0.5,
        "score_intent": False,
    },
    "medium": {
        "grid_beats": 0.5,
        "catalog": DURATION_CATALOG_MEDIUM,
        "min_rest_sec": 0.16,
        "min_rest_beats": 0.5,
        "min_note_beats": 0.5,
        "score_intent": False,
    },
    "fine": {
        "grid_beats": 0.25,
        "catalog": DURATION_CATALOG_FINE,
        "min_rest_sec": 0.08,
        "min_rest_beats": 0.2,
        "min_note_beats": 0.125,
        "score_intent": False,
    },
    # Score-oriented: reconstruct written equal subdivisions when the
    # performance looks like a continuous equal-note stream. General
    # heuristic (not piece-specific); Bach Prelude is the sandbox test.
    "score": {
        "grid_beats": 0.25,
        "catalog": DURATION_CATALOG_SCORE,
        "min_rest_sec": 0.12,
        "min_rest_beats": 0.25,
        "min_note_beats": 0.25,
        "score_intent": True,
    },
}
DEFAULT_QUANTIZE = "medium"
# Soft bias: prefer a longer catalog value when nearly as close (readable).
DURATION_LONG_BIAS = 0.12
# Under score intent, prefer the stream unit over longer near-fits.
SCORE_DURATION_LONG_BIAS = 0.04


@dataclass(frozen=True)
class RestEvent:
    """A silence gap treated as a rest in the rhythmic transcript."""

    onset: float
    offset: float
    duration: float
    kind: str = "rest"  # always "rest" (explicit for JSON consumers)


@dataclass(frozen=True)
class QuantizedRhythm:
    """Per-event rhythmic fields relative to an estimated tempo."""

    onset_beats: float
    duration_beats: float
    quantized_onset_beats: float
    quantized_duration_beats: float
    duration_value: str
    duration_value_beats: float


def resolve_quantize_preset(
    name: str | None = None,
    *,
    grid_beats: float | None = None,
    min_rest_sec: float | None = None,
) -> dict:
    """Resolve a named coarseness preset with optional CLI overrides."""
    preset_name = (name or DEFAULT_QUANTIZE).lower().strip()
    if preset_name not in QUANTIZE_PRESETS:
        raise ValueError(
            f"Unknown quantize preset '{name}'. "
            "Choose coarse|medium|fine|score."
        )
    cfg = dict(QUANTIZE_PRESETS[preset_name])
    cfg["name"] = preset_name
    if grid_beats is not None:
        cfg["grid_beats"] = float(grid_beats)
    if min_rest_sec is not None:
        cfg["min_rest_sec"] = float(min_rest_sec)
    return cfg


def detect_equal_note_stream(
    notes: Sequence[NoteEvent],
    *,
    min_notes: int = EQUAL_STREAM_MIN_NOTES,
    max_cv: float = EQUAL_STREAM_MAX_IOI_CV,
    min_ioi_sec: float = EQUAL_STREAM_MIN_IOI_SEC,
    max_ioi_sec: float = EQUAL_STREAM_MAX_IOI_SEC,
) -> dict | None:
    """Detect a continuous roughly-equal inter-onset stream (score intent).

    Returns stream stats when the monophonic onsets look like equal written
    subdivisions (moto perpetuo / prelude-style), else ``None``. Piece-agnostic:
    uses IOI statistics only — never a hard-coded melody.
    """
    if len(notes) < min_notes:
        return None
    ordered = sorted(notes, key=lambda n: (n.onset, n.offset))
    onsets = np.asarray([float(n.onset) for n in ordered], dtype=np.float64)
    iois = np.diff(onsets)
    core = iois[(iois >= min_ioi_sec) & (iois <= max_ioi_sec)]
    if core.size < max(4, min_notes // 2):
        return None
    med = float(np.median(core))
    if med <= 0:
        return None
    cv = float(np.std(core) / med)
    # Fraction of IOIs within 40% of the median (robust equal-stream vote).
    near = float(np.mean(np.abs(core - med) <= 0.40 * med))
    if cv > max_cv and near < 0.65:
        return None
    # Map median IOI to a likely written unit in beats at a provisional BPM
    # where that unit is a sixteenth / eighth / eighth-triplet-ish.
    # Prefer sixteenth when IOI is short; eighth when longer.
    if med <= 0.28:
        unit_beats = 0.25  # sixteenth
    elif med <= 0.55:
        unit_beats = 0.5  # eighth
    else:
        unit_beats = 1.0  # quarter
    bpm = 60.0 * unit_beats / med
    bpm = _clamp_bpm(bpm, min_bpm=DEFAULT_MIN_BPM, max_bpm=DEFAULT_MAX_BPM)
    return {
        "detected": True,
        "unit_sec": round(med, 4),
        "unit_beats": unit_beats,
        "ioi_cv": round(cv, 4),
        "ioi_near_frac": round(near, 4),
        "num_iois": int(core.size),
        "suggested_bpm": round(float(bpm), 2),
        "method": "equal_note_stream",
    }


def _snap_duration_to_stream_unit(
    duration_sec: float,
    *,
    unit_sec: float,
    unit_beats: float,
    catalog: Sequence[tuple[float, str]],
    hold_mult: float = EQUAL_STREAM_HOLD_MULT,
) -> tuple[str, float]:
    """Map a sounding length to written equal-subdivision intent.

    Near-unit notes → exactly one unit (so held open-string flavour still
    reads as a written sixteenth/eighth). Clearly longer notes → integer
    multiples of the unit, snapped to the catalog.
    """
    if unit_sec <= 0 or duration_sec <= 0:
        return nearest_duration_value(
            unit_beats, catalog=catalog, min_beats=unit_beats, long_bias=0.0
        )
    ratio = float(duration_sec) / float(unit_sec)
    if ratio < hold_mult:
        n_units = 1
    else:
        n_units = max(1, int(round(ratio)))
    target_beats = n_units * unit_beats
    return nearest_duration_value(
        target_beats,
        catalog=catalog,
        min_beats=unit_beats,
        long_bias=SCORE_DURATION_LONG_BIAS,
    )


def _clamp_bpm(bpm: float, *, min_bpm: float, max_bpm: float) -> float:
    if not np.isfinite(bpm) or bpm <= 0:
        return 120.0
    while bpm < min_bpm:
        bpm *= 2.0
    while bpm > max_bpm:
        bpm /= 2.0
    return float(np.clip(bpm, min_bpm, max_bpm))


def _ioi_to_bpm(
    iois: np.ndarray,
    *,
    min_bpm: float,
    max_bpm: float,
) -> float | None:
    if iois.size == 0:
        return None
    med = float(np.median(iois))
    if med <= 0:
        return None
    return _clamp_bpm(60.0 / med, min_bpm=min_bpm, max_bpm=max_bpm)


def _tempo_from_note_onsets(
    onsets: Sequence[float],
    *,
    min_bpm: float,
    max_bpm: float,
) -> float | None:
    """Median inter-onset interval → BPM (monophonic melody heuristic)."""
    if len(onsets) < 2:
        return None
    iois = np.diff(np.asarray(onsets, dtype=np.float64))
    iois = iois[(iois >= MIN_TEMPO_IOI_SEC) & (iois <= MAX_TEMPO_IOI_SEC)]
    return _ioi_to_bpm(iois, min_bpm=min_bpm, max_bpm=max_bpm)


def _tempo_from_note_durations(
    durations: Sequence[float],
    *,
    min_bpm: float,
    max_bpm: float,
    max_beat_sec: float = 0.85,
) -> float | None:
    """Median sounding-note length → BPM, assuming notes ≈ one beat.

    More stable than onset IOIs when intentional rests sit between notes
    (IOI then spans note+rest = two beats). Ignores short vibrato flicker
    and very long sustains (multi-beat notes would read an octave low).
    """
    durs = np.asarray(
        [
            d
            for d in durations
            if d is not None and MIN_TEMPO_NOTE_DUR_SEC <= float(d) <= max_beat_sec
        ],
        dtype=np.float64,
    )
    if durs.size == 0:
        return None
    med = float(np.median(durs))
    if med <= 0:
        return None
    return _clamp_bpm(60.0 / med, min_bpm=min_bpm, max_bpm=max_bpm)


def _tempo_from_spectral_onsets(
    audio: np.ndarray,
    *,
    sr: int,
    hop_length: int,
    min_bpm: float,
    max_bpm: float,
) -> tuple[float | None, list[float]]:
    """BPM from spectral-flux onset peaks (ignores pitch-flicker note splits)."""
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=hop_length
    )
    wait = max(1, int(round(0.10 * sr / hop_length)))
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
        wait=wait,
        delta=0.08,
        pre_max=wait,
        post_max=wait,
        pre_avg=wait * 2,
        post_avg=wait * 2,
    )
    times = librosa.frames_to_time(
        np.asarray(frames, dtype=np.int64), sr=sr, hop_length=hop_length
    )
    times_list = [round(float(t), 4) for t in times]
    if len(times_list) < 3:
        return None, times_list
    iois = np.diff(np.asarray(times_list, dtype=np.float64))
    iois = iois[(iois >= MIN_TEMPO_IOI_SEC) & (iois <= MAX_TEMPO_IOI_SEC)]
    return _ioi_to_bpm(iois, min_bpm=min_bpm, max_bpm=max_bpm), times_list


def _duration_fit_score(
    durations_sec: Sequence[float],
    bpm: float,
    *,
    catalog: Sequence[tuple[float, str]] = DURATION_CATALOG_MEDIUM,
    max_score_beats: float = 4.0,
) -> float:
    """Lower is better: mean absolute error of durations vs nearest catalog value.

    Notes longer than ``max_score_beats`` at the candidate tempo are capped
    so multi-beat sustains do not pull octave resolve toward half-tempo.
    """
    durs = [
        float(d)
        for d in durations_sec
        if d is not None and float(d) >= MIN_TEMPO_NOTE_DUR_SEC
    ]
    if not durs or bpm <= 0:
        return 1e9
    errs: list[float] = []
    for d in durs:
        beats = seconds_to_beats(float(d), bpm)
        # Cap before nearest-catalog so a 6-beat sustain @122 does not
        # score worse than a 3-beat reading @61.
        scored = min(beats, max_score_beats)
        _, nearest = nearest_duration_value(
            scored, catalog=catalog, long_bias=0.0
        )
        errs.append(abs(scored - nearest))
    return float(np.mean(errs))


def _pick_bpm_with_octave_fix(
    seed_bpm: float,
    durations_sec: Sequence[float],
    *,
    min_bpm: float,
    max_bpm: float,
    prefer_midrange: bool = True,
    catalog: Sequence[tuple[float, str]] = DURATION_CATALOG_MEDIUM,
    spectral_hint: float | None = None,
    extra_candidates: Sequence[tuple[str, float]] | None = None,
) -> tuple[float, str]:
    """Choose among bpm, 2·bpm, ½·bpm (plus extras) by catalog duration fit.

    Octave ambiguity is common for short monophonic clips. Prefer the
    candidate whose note lengths land closest to named values, with a soft
    bias toward practice tempi (~70–140) and any spectral/duration hint.
    """
    candidates: list[tuple[float, str]] = []
    for factor, tag in ((1.0, "seed"), (2.0, "double"), (0.5, "half")):
        cand = _clamp_bpm(seed_bpm * factor, min_bpm=min_bpm, max_bpm=max_bpm)
        if any(abs(cand - c) < 0.5 for c, _ in candidates):
            continue
        candidates.append((cand, tag))
    if extra_candidates:
        for tag, raw in extra_candidates:
            if raw is None or raw <= 0:
                continue
            cand = _clamp_bpm(float(raw), min_bpm=min_bpm, max_bpm=max_bpm)
            if any(abs(cand - c) < 0.5 for c, _ in candidates):
                continue
            candidates.append((cand, tag))

    scored = [
        (_duration_fit_score(durations_sec, c, catalog=catalog), c, tag)
        for c, tag in candidates
    ]
    seed_err = next(e for e, _, tag in scored if tag == "seed")
    IMPROVE = 0.05
    viable = [
        (e, c, tag)
        for e, c, tag in scored
        if tag == "seed" or e <= seed_err - IMPROVE
    ]

    def _rank_key(item: tuple[float, float, str]) -> tuple:
        err, bpm, _tag = item
        mid = 0.0
        if prefer_midrange and not (70.0 <= bpm <= 140.0):
            mid = 0.05
        # Pull toward hint tempo when present.
        hint = 0.0
        if spectral_hint is not None and spectral_hint > 0:
            hint = 0.04 * abs(np.log2(max(bpm, 1.0) / spectral_hint))
        return (err + mid + hint, abs(np.log2(max(bpm, 1.0) / 120.0)))

    viable.sort(key=_rank_key)
    best_err, best_bpm, best_tag = viable[0]
    return best_bpm, f"octave_resolve:{best_tag}(err={best_err:.3f})"


def estimate_tempo(
    audio: np.ndarray,
    *,
    sr: int,
    note_onsets: Sequence[float] | None = None,
    note_durations: Sequence[float] | None = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
    bpm_override: float | None = None,
) -> dict:
    """Estimate BPM + beat times via spectral onsets, refined with notes.

    Returns a dict suitable for JSON (bpm, beat_times, method, …).
    ``bpm_override`` skips estimation (CLI ``--bpm``).
    """
    if bpm_override is not None and bpm_override > 0:
        bpm = _clamp_bpm(float(bpm_override), min_bpm=min_bpm, max_bpm=max_bpm)
        beat_period = 60.0 / bpm
        duration_s = float(len(audio) / sr)
        anchor = float(note_onsets[0]) if note_onsets else 0.0
        grid: list[float] = []
        t = anchor
        while t - beat_period >= -1e-6:
            t -= beat_period
        while t <= duration_s + 1e-6:
            if t >= -1e-6:
                grid.append(round(float(t), 4))
            t += beat_period
        return {
            "bpm": round(float(bpm), 2),
            "beat_period_sec": round(beat_period, 4),
            "method": "cli_override",
            "audio_bpm": None,
            "spectral_onset_bpm": None,
            "note_ioi_bpm": None,
            "note_duration_bpm": None,
            "librosa_beat_times": [],
            "spectral_onset_times": [],
            "beat_times": grid,
            "num_beats": len(grid),
        }

    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=hop_length
    )
    tempo_arr, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        units="frames",
    )
    audio_bpm = float(np.atleast_1d(tempo_arr)[0])
    audio_bpm = _clamp_bpm(audio_bpm, min_bpm=min_bpm, max_bpm=max_bpm)

    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=hop_length
    )
    beat_times_list = [round(float(t), 4) for t in np.asarray(beat_times)]

    spectral_bpm, spectral_times = _tempo_from_spectral_onsets(
        audio,
        sr=sr,
        hop_length=hop_length,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )

    ioi_bpm = None
    if note_onsets is not None:
        ioi_bpm = _tempo_from_note_onsets(
            note_onsets, min_bpm=min_bpm, max_bpm=max_bpm
        )
    dur_bpm = None
    if note_durations is not None:
        dur_bpm = _tempo_from_note_durations(
            note_durations, min_bpm=min_bpm, max_bpm=max_bpm
        )

    method = "librosa.beat.beat_track"
    bpm = audio_bpm

    # Spectral onset IOI is the primary pulse for real cello phrases
    # (stable vs vibrato note-splits and multi-beat sustains). Duration
    # tempo only wins when spectral is missing / out of range and the
    # note lengths look like single-beat articulations (rest-gap clips).
    def _octaves_apart(a: float, b: float) -> float:
        return abs(np.log2(max(a, 1e-6) / max(b, 1e-6)))

    def _in_midrange(v: float) -> bool:
        return 70.0 <= v <= 140.0

    if spectral_bpm is not None and _in_midrange(spectral_bpm):
        if dur_bpm is not None and _octaves_apart(dur_bpm, spectral_bpm) < 0.25:
            bpm = 0.45 * spectral_bpm + 0.55 * dur_bpm
            method = "blend_spectral_note_duration"
        else:
            ratio = max(audio_bpm, spectral_bpm) / max(
                1e-6, min(audio_bpm, spectral_bpm)
            )
            if ratio > 1.2:
                bpm = spectral_bpm
                method = "spectral_onset_ioi"
            else:
                bpm = (
                    (1.0 - IOI_TEMPO_WEIGHT) * audio_bpm
                    + IOI_TEMPO_WEIGHT * spectral_bpm
                )
                method = "blend_beat_track_spectral_onset"
        bpm = _clamp_bpm(bpm, min_bpm=min_bpm, max_bpm=max_bpm)
    elif dur_bpm is not None and (
        spectral_bpm is None
        or not _in_midrange(spectral_bpm)
        or _octaves_apart(dur_bpm, spectral_bpm) > 0.35
    ):
        if spectral_bpm is not None and _octaves_apart(dur_bpm, spectral_bpm) < 0.25:
            bpm = 0.35 * spectral_bpm + 0.65 * dur_bpm
            method = "blend_spectral_note_duration"
        elif _octaves_apart(audio_bpm, dur_bpm) < 0.3:
            bpm = 0.35 * audio_bpm + 0.65 * dur_bpm
            method = "blend_beat_track_note_duration"
        else:
            bpm = dur_bpm
            method = "note_duration_tempo"
        bpm = _clamp_bpm(bpm, min_bpm=min_bpm, max_bpm=max_bpm)
    elif spectral_bpm is not None:
        bpm = spectral_bpm
        method = "spectral_onset_ioi"
        bpm = _clamp_bpm(bpm, min_bpm=min_bpm, max_bpm=max_bpm)
    elif ioi_bpm is not None:
        ratio = max(bpm, ioi_bpm) / max(1e-6, min(bpm, ioi_bpm))
        if ratio < 1.35 or len(beat_times_list) < 4:
            bpm = (
                (1.0 - IOI_TEMPO_WEIGHT) * audio_bpm
                + IOI_TEMPO_WEIGHT * ioi_bpm
            )
            method = "blend_beat_track_note_ioi"
        bpm = _clamp_bpm(bpm, min_bpm=min_bpm, max_bpm=max_bpm)
    elif dur_bpm is not None and len(beat_times_list) < 4:
        bpm = 0.35 * audio_bpm + 0.65 * dur_bpm
        method = "blend_beat_track_note_duration"
        bpm = _clamp_bpm(bpm, min_bpm=min_bpm, max_bpm=max_bpm)

    # Resolve half/double tempo using sounding-note duration fit.
    # Prefer spectral as the octave hint when it sits in a practice range.
    if note_durations:
        hint = (
            spectral_bpm
            if (spectral_bpm is not None and _in_midrange(spectral_bpm))
            else dur_bpm
        )
        extras: list[tuple[str, float]] = []
        if dur_bpm is not None:
            extras.append(("duration", float(dur_bpm)))
        if spectral_bpm is not None:
            extras.append(("spectral", float(spectral_bpm)))
        resolved, resolve_tag = _pick_bpm_with_octave_fix(
            bpm,
            note_durations,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            spectral_hint=hint,
            extra_candidates=extras or None,
        )
        bpm = resolved
        method = f"{method}+{resolve_tag}"

    beat_period = 60.0 / bpm
    if beat_times_list:
        anchor = beat_times_list[0]
    elif spectral_times:
        anchor = float(spectral_times[0])
    elif note_onsets:
        anchor = float(note_onsets[0])
    else:
        anchor = 0.0

    duration_s = float(len(audio) / sr)
    grid = []
    t = anchor
    while t - beat_period >= -1e-6:
        t -= beat_period
    while t <= duration_s + 1e-6:
        if t >= -1e-6:
            grid.append(round(float(t), 4))
        t += beat_period

    return {
        "bpm": round(float(bpm), 2),
        "beat_period_sec": round(beat_period, 4),
        "method": method,
        "audio_bpm": round(float(audio_bpm), 2),
        "spectral_onset_bpm": (
            round(float(spectral_bpm), 2) if spectral_bpm is not None else None
        ),
        "note_ioi_bpm": round(float(ioi_bpm), 2) if ioi_bpm is not None else None,
        "note_duration_bpm": (
            round(float(dur_bpm), 2) if dur_bpm is not None else None
        ),
        "librosa_beat_times": beat_times_list,
        "spectral_onset_times": spectral_times,
        "beat_times": grid,
        "num_beats": len(grid),
    }


def seconds_to_beats(seconds: float, bpm: float) -> float:
    return float(seconds) * float(bpm) / 60.0


def beats_to_seconds(beats: float, bpm: float) -> float:
    return float(beats) * 60.0 / float(bpm)


def quantize_beats(
    value: float,
    *,
    grid: float = DEFAULT_GRID_BEATS,
) -> float:
    """Snap a beat-position or length to the nearest grid subdivision."""
    if grid <= 0:
        return float(value)
    return float(round(value / grid) * grid)


def nearest_duration_value(
    duration_beats: float,
    *,
    catalog: Sequence[tuple[float, str]] = DURATION_CATALOG_MEDIUM,
    min_beats: float = 0.0,
    long_bias: float = DURATION_LONG_BIAS,
) -> tuple[str, float]:
    """Map a duration in beats to the nearest named note/rest value.

    Prefers longer / coarser catalog entries when the absolute error is
    within ``long_bias`` of the tightest fit — fewer 16th/tied fragments.
    """
    if duration_beats <= 0:
        return "zero", 0.0
    target = max(float(duration_beats), float(min_beats))
    candidates = [
        (beats, name)
        for beats, name in catalog
        if beats + 1e-9 >= min_beats
    ]
    if not candidates:
        beats, name = catalog[-1]
        return name, float(beats)

    best_err = min(abs(target - beats) for beats, _ in candidates)
    viable = [
        (beats, name)
        for beats, name in candidates
        if abs(target - beats) <= best_err + long_bias + 1e-9
    ]
    # Longest first; on equal length, closer fit; stable for ties.
    viable.sort(key=lambda bn: (-bn[0], abs(target - bn[0])))
    best_beats, best_name = viable[0]
    return best_name, float(best_beats)


def quantize_interval(
    onset_sec: float,
    duration_sec: float,
    *,
    bpm: float,
    grid: float = DEFAULT_GRID_BEATS,
    tempo_anchor_sec: float = 0.0,
    catalog: Sequence[tuple[float, str]] = DURATION_CATALOG_MEDIUM,
    min_beats: float = DEFAULT_MIN_NOTE_BEATS,
) -> QuantizedRhythm:
    """Convert a time interval into beat-relative quantized fields."""
    onset_beats = seconds_to_beats(onset_sec - tempo_anchor_sec, bpm)
    duration_beats = seconds_to_beats(duration_sec, bpm)
    q_onset = quantize_beats(onset_beats, grid=grid)
    name, q_dur = nearest_duration_value(
        max(duration_beats, 0.0), catalog=catalog, min_beats=min_beats
    )
    if duration_beats > 0 and q_dur <= 0:
        name, q_dur = nearest_duration_value(
            max(grid, min_beats), catalog=catalog, min_beats=min_beats
        )
    return QuantizedRhythm(
        onset_beats=round(onset_beats, 4),
        duration_beats=round(duration_beats, 4),
        quantized_onset_beats=round(q_onset, 4),
        quantized_duration_beats=round(q_dur, 4),
        duration_value=name,
        duration_value_beats=q_dur,
    )


def extract_rests(
    notes: Sequence[NoteEvent],
    *,
    audio_duration_sec: float | None = None,
    min_rest_sec: float = DEFAULT_MIN_REST_SEC,
    include_leading: bool = True,
    include_trailing: bool = True,
) -> list[RestEvent]:
    """Build rest events from silence gaps around / between notes.

    Inter-note gaps are the primary rest signal (preference: rhythm must
    include rests, not only sounding notes). Leading/trailing silence is
    included when long enough so padded clips still surface bookend rests.
    """
    rests: list[RestEvent] = []
    if not notes:
        if (
            audio_duration_sec is not None
            and audio_duration_sec >= min_rest_sec
        ):
            rests.append(
                RestEvent(
                    onset=0.0,
                    offset=round(float(audio_duration_sec), 4),
                    duration=round(float(audio_duration_sec), 4),
                )
            )
        return rests

    ordered = sorted(notes, key=lambda n: (n.onset, n.offset))

    if include_leading and ordered[0].onset >= min_rest_sec:
        onset = 0.0
        offset = float(ordered[0].onset)
        rests.append(
            RestEvent(
                onset=round(onset, 4),
                offset=round(offset, 4),
                duration=round(offset - onset, 4),
            )
        )

    for a, b in zip(ordered, ordered[1:]):
        gap = float(b.onset) - float(a.offset)
        if gap >= min_rest_sec:
            rests.append(
                RestEvent(
                    onset=round(float(a.offset), 4),
                    offset=round(float(b.onset), 4),
                    duration=round(gap, 4),
                )
            )

    if (
        include_trailing
        and audio_duration_sec is not None
        and (audio_duration_sec - ordered[-1].offset) >= min_rest_sec
    ):
        onset = float(ordered[-1].offset)
        offset = float(audio_duration_sec)
        rests.append(
            RestEvent(
                onset=round(onset, 4),
                offset=round(offset, 4),
                duration=round(offset - onset, 4),
            )
        )

    return rests


def _rhythm_fields(q: QuantizedRhythm) -> dict:
    return {
        "onset_beats": q.onset_beats,
        "duration_beats": q.duration_beats,
        "quantized_onset_beats": q.quantized_onset_beats,
        "quantized_duration_beats": q.quantized_duration_beats,
        "duration_value": q.duration_value,
        "duration_value_beats": q.duration_value_beats,
    }


def analyze_rhythm(
    audio: np.ndarray,
    notes: Sequence[NoteEvent],
    *,
    sr: int,
    hop_length: int = 512,
    grid_beats: float = DEFAULT_GRID_BEATS,
    min_rest_sec: float = DEFAULT_MIN_REST_SEC,
    min_rest_beats: float = DEFAULT_MIN_REST_BEATS,
    min_note_beats: float = DEFAULT_MIN_NOTE_BEATS,
    catalog: Sequence[tuple[float, str]] = DURATION_CATALOG_MEDIUM,
    include_leading_rests: bool = True,
    include_trailing_rests: bool = True,
    bpm_override: float | None = None,
    score_intent: bool = False,
) -> dict:
    """Full rhythm layer: tempo + quantized notes + quantized rests.

    When ``score_intent`` is True and an equal-note stream is detected,
    durations snap to the written subdivision unit (performance holds /
    rubato still read as equal sixteenths/eighths). Heuristic is general
    for monophonic cello — not hard-coded to any piece.

    Returns a JSON-ready dict with keys ``tempo``, ``notes`` (rhythm
    fields only, parallel to input order), ``rests``, and ``events``
    (chronological note/rest stream for consumers that want one timeline).
    """
    note_onsets = [float(n.onset) for n in notes]
    note_durations = [float(n.duration) for n in notes]

    stream = detect_equal_note_stream(notes) if score_intent else None
    effective_bpm_override = bpm_override
    if (
        score_intent
        and stream is not None
        and (bpm_override is None or bpm_override <= 0)
    ):
        # Prefer stream-implied tempo (written pulse) over raw beat_track.
        effective_bpm_override = float(stream["suggested_bpm"])

    tempo = estimate_tempo(
        audio,
        sr=sr,
        note_onsets=note_onsets,
        note_durations=note_durations,
        hop_length=hop_length,
        bpm_override=effective_bpm_override,
    )
    if stream is not None and bpm_override is None:
        tempo = dict(tempo)
        tempo["equal_stream"] = stream
        tempo["method"] = f"{tempo.get('method', '')}+equal_stream".strip("+")
        if effective_bpm_override is not None:
            tempo["method"] = (
                f"equal_note_stream(unit={stream['unit_beats']},"
                f"cv={stream['ioi_cv']})"
            )
            tempo["bpm"] = float(stream["suggested_bpm"])
            # Rebuild beat grid at stream BPM anchored on first note.
            beat_period = 60.0 / float(tempo["bpm"])
            tempo["beat_period_sec"] = round(beat_period, 4)
            duration_s = float(len(audio) / sr)
            anchor0 = float(note_onsets[0]) if note_onsets else 0.0
            grid: list[float] = []
            t = anchor0
            while t - beat_period >= -1e-6:
                t -= beat_period
            while t <= duration_s + 1e-6:
                if t >= -1e-6:
                    grid.append(round(float(t), 4))
                t += beat_period
            tempo["beat_times"] = grid
            tempo["num_beats"] = len(grid)

    bpm = float(tempo["bpm"])
    anchor = float(tempo["beat_times"][0]) if tempo["beat_times"] else (
        note_onsets[0] if note_onsets else 0.0
    )

    # Score-intent grid: snap onsets to the stream unit when detected.
    effective_grid = grid_beats
    effective_min_note = min_note_beats
    effective_min_rest_sec = min_rest_sec
    effective_min_rest_beats = min_rest_beats
    long_bias = DURATION_LONG_BIAS
    if stream is not None:
        effective_grid = float(stream["unit_beats"])
        effective_min_note = float(stream["unit_beats"])
        # Suppress articulation micro-gaps so continuous streams stay rest-free.
        effective_min_rest_sec = max(
            min_rest_sec,
            float(stream["unit_sec"]) * EQUAL_STREAM_REST_SUPPRESS_MULT,
        )
        effective_min_rest_beats = float(stream["unit_beats"])

    note_rhythm: list[dict] = []
    for n in notes:
        if stream is not None:
            onset_beats = seconds_to_beats(float(n.onset) - anchor, bpm)
            q_onset = quantize_beats(onset_beats, grid=effective_grid)
            name, q_dur = _snap_duration_to_stream_unit(
                float(n.duration),
                unit_sec=float(stream["unit_sec"]),
                unit_beats=float(stream["unit_beats"]),
                catalog=catalog,
            )
            duration_beats = seconds_to_beats(float(n.duration), bpm)
            q = QuantizedRhythm(
                onset_beats=round(onset_beats, 4),
                duration_beats=round(duration_beats, 4),
                quantized_onset_beats=round(q_onset, 4),
                quantized_duration_beats=round(q_dur, 4),
                duration_value=name,
                duration_value_beats=q_dur,
            )
        else:
            q = quantize_interval(
                float(n.onset),
                float(n.duration),
                bpm=bpm,
                grid=effective_grid,
                tempo_anchor_sec=anchor,
                catalog=catalog,
                min_beats=effective_min_note,
            )
        note_rhythm.append(_rhythm_fields(q))

    raw_rests = extract_rests(
        notes,
        audio_duration_sec=float(len(audio) / sr),
        min_rest_sec=effective_min_rest_sec,
        include_leading=include_leading_rests,
        include_trailing=include_trailing_rests,
    )

    rests_out: list[dict] = []
    for r in raw_rests:
        q = quantize_interval(
            float(r.onset),
            float(r.duration),
            bpm=bpm,
            grid=effective_grid,
            tempo_anchor_sec=anchor,
            catalog=catalog,
            min_beats=effective_min_rest_beats,
        )
        if (
            q.quantized_duration_beats < effective_min_rest_beats
            and q.duration_beats < effective_min_rest_beats
        ):
            continue
        entry = asdict(r)
        entry.update(_rhythm_fields(q))
        rests_out.append(entry)

    events: list[dict] = []
    for n, nr in zip(notes, note_rhythm):
        events.append(
            {
                "type": "note",
                "midi": n.midi,
                "note": n.note,
                "onset": n.onset,
                "offset": n.offset,
                "duration": n.duration,
                "confidence": n.confidence,
                **nr,
            }
        )
    for r in rests_out:
        events.append({"type": "rest", **r})
    events.sort(key=lambda e: (float(e["onset"]), 0 if e["type"] == "note" else 1))

    return {
        "tempo": tempo,
        "grid_beats": effective_grid,
        "min_rest_sec": effective_min_rest_sec,
        "min_rest_beats": effective_min_rest_beats,
        "min_note_beats": effective_min_note,
        "score_intent": bool(score_intent),
        "equal_stream": stream,
        "note_rhythm": note_rhythm,
        "rests": rests_out,
        "num_rests": len(rests_out),
        "events": events,
    }
