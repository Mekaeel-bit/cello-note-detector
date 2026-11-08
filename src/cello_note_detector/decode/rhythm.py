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

# Back-compat alias (fine catalog).
DURATION_CATALOG = DURATION_CATALOG_FINE

# Default snap grid: eighth note (was sixteenth). Readable over fussy.
DEFAULT_GRID_BEATS = 0.5
DEFAULT_MIN_BPM = 50.0
DEFAULT_MAX_BPM = 180.0
DEFAULT_MIN_REST_SEC = 0.12
DEFAULT_MIN_REST_BEATS = 0.4
DEFAULT_MIN_NOTE_BEATS = 0.5
# Prefer note-onset / spectral IOI tempo when beat_track disagrees.
IOI_TEMPO_WEIGHT = 0.7
# Ignore sub-articulation IOIs when estimating tempo from onsets.
MIN_TEMPO_IOI_SEC = 0.18
MAX_TEMPO_IOI_SEC = 1.8
# Ignore vibrato flicker when scoring duration-vs-BPM fit.
MIN_TEMPO_NOTE_DUR_SEC = 0.15

QUANTIZE_PRESETS: dict[str, dict] = {
    "coarse": {
        "grid_beats": 0.5,
        "catalog": DURATION_CATALOG_COARSE,
        "min_rest_sec": 0.15,
        "min_rest_beats": 0.5,
        "min_note_beats": 0.5,
    },
    "medium": {
        "grid_beats": 0.5,
        "catalog": DURATION_CATALOG_MEDIUM,
        "min_rest_sec": 0.12,
        "min_rest_beats": 0.4,
        "min_note_beats": 0.5,
    },
    "fine": {
        "grid_beats": 0.25,
        "catalog": DURATION_CATALOG_FINE,
        "min_rest_sec": 0.08,
        "min_rest_beats": 0.2,
        "min_note_beats": 0.125,
    },
}
DEFAULT_QUANTIZE = "medium"


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
            f"Unknown quantize preset '{name}'. Choose coarse|medium|fine."
        )
    cfg = dict(QUANTIZE_PRESETS[preset_name])
    cfg["name"] = preset_name
    if grid_beats is not None:
        cfg["grid_beats"] = float(grid_beats)
    if min_rest_sec is not None:
        cfg["min_rest_sec"] = float(min_rest_sec)
    return cfg


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
) -> float | None:
    """Median sounding-note length → BPM, assuming notes ≈ one beat.

    More stable than onset IOIs when intentional rests sit between notes
    (IOI then spans note+rest = two beats). Ignores short vibrato flicker.
    """
    durs = np.asarray(
        [
            d
            for d in durations
            if d is not None and d >= MIN_TEMPO_NOTE_DUR_SEC
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
) -> float:
    """Lower is better: mean absolute error of durations vs nearest catalog value."""
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
        _, nearest = nearest_duration_value(beats, catalog=catalog)
        errs.append(abs(beats - nearest))
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

    # When intentional rests sit between notes, spectral/note IOIs often
    # span note+rest (= 2 beats) and read an octave low. Prefer sounding-
    # note duration tempo in that case. Otherwise spectral onsets beat
    # vibrato-contaminated note-split IOIs.
    def _octaves_apart(a: float, b: float) -> float:
        return abs(np.log2(max(a, 1e-6) / max(b, 1e-6)))

    duration_vs_spectral = (
        dur_bpm is not None
        and spectral_bpm is not None
        and _octaves_apart(dur_bpm, spectral_bpm) > 0.35
    )
    duration_vs_ioi = (
        dur_bpm is not None
        and ioi_bpm is not None
        and _octaves_apart(dur_bpm, ioi_bpm) > 0.35
    )

    if dur_bpm is not None and (duration_vs_spectral or duration_vs_ioi or spectral_bpm is None):
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
    # Also consider duration/spectral estimates as extra candidates so a
    # half-tempo seed can still jump to the true pulse.
    if note_durations:
        resolved, resolve_tag = _pick_bpm_with_octave_fix(
            bpm,
            note_durations,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            spectral_hint=spectral_bpm if not duration_vs_spectral else dur_bpm,
            extra_candidates=(
                [("duration", dur_bpm)]
                if dur_bpm is not None
                else None
            ),
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
) -> tuple[str, float]:
    """Map a duration in beats to the nearest named note/rest value."""
    if duration_beats <= 0:
        return "zero", 0.0
    target = max(float(duration_beats), float(min_beats))
    best_name = catalog[-1][1]
    best_beats = catalog[-1][0]
    best_err = abs(target - best_beats)
    for beats, name in catalog:
        if beats + 1e-9 < min_beats:
            continue
        err = abs(target - beats)
        # Prefer closer; on ties prefer the catalog entry (coarser first).
        if err < best_err - 1e-9:
            best_err = err
            best_name = name
            best_beats = beats
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
) -> dict:
    """Full rhythm layer: tempo + quantized notes + quantized rests.

    Returns a JSON-ready dict with keys ``tempo``, ``notes`` (rhythm
    fields only, parallel to input order), ``rests``, and ``events``
    (chronological note/rest stream for consumers that want one timeline).
    """
    note_onsets = [float(n.onset) for n in notes]
    note_durations = [float(n.duration) for n in notes]
    tempo = estimate_tempo(
        audio,
        sr=sr,
        note_onsets=note_onsets,
        note_durations=note_durations,
        hop_length=hop_length,
        bpm_override=bpm_override,
    )
    bpm = float(tempo["bpm"])
    anchor = float(tempo["beat_times"][0]) if tempo["beat_times"] else (
        note_onsets[0] if note_onsets else 0.0
    )

    note_rhythm: list[dict] = []
    for n in notes:
        q = quantize_interval(
            float(n.onset),
            float(n.duration),
            bpm=bpm,
            grid=grid_beats,
            tempo_anchor_sec=anchor,
            catalog=catalog,
            min_beats=min_note_beats,
        )
        note_rhythm.append(_rhythm_fields(q))

    raw_rests = extract_rests(
        notes,
        audio_duration_sec=float(len(audio) / sr),
        min_rest_sec=min_rest_sec,
        include_leading=include_leading_rests,
        include_trailing=include_trailing_rests,
    )

    rests_out: list[dict] = []
    for r in raw_rests:
        q = quantize_interval(
            float(r.onset),
            float(r.duration),
            bpm=bpm,
            grid=grid_beats,
            tempo_anchor_sec=anchor,
            catalog=catalog,
            min_beats=min_rest_beats,
        )
        if (
            q.quantized_duration_beats < min_rest_beats
            and q.duration_beats < min_rest_beats
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
        "grid_beats": grid_beats,
        "min_rest_sec": min_rest_sec,
        "min_rest_beats": min_rest_beats,
        "min_note_beats": min_note_beats,
        "note_rhythm": note_rhythm,
        "rests": rests_out,
        "num_rests": len(rests_out),
        "events": events,
    }
