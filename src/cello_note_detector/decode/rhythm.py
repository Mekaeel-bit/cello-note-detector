"""Tempo estimation, beat grid, duration quantization, and rest gaps.

Classical DSP path (librosa beat tracking + note-onset IOIs). Emits
quantized rhythmic metadata for sounding notes and for silences between
them (rests) — sheet music needs both.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import librosa
import numpy as np

from cello_note_detector.decode.events import NoteEvent

# Duration catalog in beats (quarter note = 1.0), coarsest → finest.
# Names follow common American English note-value vocabulary.
DURATION_CATALOG: tuple[tuple[float, str], ...] = (
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

# Snap onsets/durations to this beat subdivision (sixteenth = 0.25).
DEFAULT_GRID_BEATS = 0.25
DEFAULT_MIN_BPM = 40.0
DEFAULT_MAX_BPM = 200.0
# Absolute floor so tiny pyin gaps / bow noise do not become rests.
DEFAULT_MIN_REST_SEC = 0.08
# Ignore rests shorter than this fraction of a beat after quantization.
DEFAULT_MIN_REST_BEATS = 0.2
# Prefer note-onset IOI tempo when beat_track disagrees wildly.
IOI_TEMPO_WEIGHT = 0.65


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


def _clamp_bpm(bpm: float, *, min_bpm: float, max_bpm: float) -> float:
    if not np.isfinite(bpm) or bpm <= 0:
        return 120.0
    while bpm < min_bpm:
        bpm *= 2.0
    while bpm > max_bpm:
        bpm /= 2.0
    return float(np.clip(bpm, min_bpm, max_bpm))


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
    iois = iois[iois > 0.05]
    if iois.size == 0:
        return None
    # Prefer IOIs that look like note lengths, not tiny re-attacks.
    med = float(np.median(iois))
    if med <= 0:
        return None
    # Assume median IOI ≈ one beat (quarter) for stepwise phrases.
    bpm = 60.0 / med
    return _clamp_bpm(bpm, min_bpm=min_bpm, max_bpm=max_bpm)


def _tempo_from_note_durations(
    durations: Sequence[float],
    *,
    min_bpm: float,
    max_bpm: float,
) -> float | None:
    """Median sounding-note length → BPM, assuming notes ≈ one beat.

    More stable than onset IOIs when intentional rests sit between notes
    (IOI then spans note+rest = two beats).
    """
    durs = np.asarray(
        [d for d in durations if d is not None and d > 0.08],
        dtype=np.float64,
    )
    if durs.size == 0:
        return None
    med = float(np.median(durs))
    if med <= 0:
        return None
    return _clamp_bpm(60.0 / med, min_bpm=min_bpm, max_bpm=max_bpm)


def _duration_fit_score(
    durations_sec: Sequence[float],
    bpm: float,
    *,
    catalog: Sequence[tuple[float, str]] = DURATION_CATALOG,
) -> float:
    """Lower is better: mean absolute error of durations vs nearest catalog value."""
    if not durations_sec or bpm <= 0:
        return 1e9
    errs: list[float] = []
    for d in durations_sec:
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
) -> tuple[float, str]:
    """Choose among bpm, 2·bpm, ½·bpm by catalog fit of note durations.

    Octave ambiguity is common for short monophonic clips (especially when
    rests make inter-onset intervals span two beats). Prefer the candidate
    whose note lengths land closest to named values (quarter/eighth/…).
    Only flip octave when the fit improves by a clear margin — slight pyin
    trimming must not drag a clean ~100 BPM estimate down to ~50.
    """
    candidates: list[tuple[float, str]] = []
    for factor, tag in ((1.0, "seed"), (2.0, "double"), (0.5, "half")):
        cand = _clamp_bpm(seed_bpm * factor, min_bpm=min_bpm, max_bpm=max_bpm)
        # Deduplicate near-equals.
        if any(abs(cand - c) < 0.5 for c, _ in candidates):
            continue
        candidates.append((cand, tag))

    scored = [
        (_duration_fit_score(durations_sec, c), c, tag) for c, tag in candidates
    ]
    seed_err = next(e for e, _, tag in scored if tag == "seed")
    # Significant improvement required to leave the seed tempo.
    IMPROVE = 0.08
    viable = [
        (e, c, tag)
        for e, c, tag in scored
        if tag == "seed" or e <= seed_err - IMPROVE
    ]

    def _rank_key(item: tuple[float, float, str]) -> tuple:
        err, bpm, _tag = item
        # Soft preference for classical practice tempi.
        mid = 0.0
        if prefer_midrange and not (70.0 <= bpm <= 140.0):
            mid = 0.03
        return (err + mid, abs(np.log2(max(bpm, 1.0) / 100.0)))

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
) -> dict:
    """Estimate BPM + beat times via librosa, refined with note IOIs.

    Returns a dict suitable for JSON (bpm, beat_times, method, …).
    """
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
    # librosa ≥0.10 may return tempo as ndarray.
    audio_bpm = float(np.atleast_1d(tempo_arr)[0])
    audio_bpm = _clamp_bpm(audio_bpm, min_bpm=min_bpm, max_bpm=max_bpm)

    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=hop_length
    )
    beat_times_list = [round(float(t), 4) for t in np.asarray(beat_times)]

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
    # Prefer sounding-note duration tempo when rests make IOIs span 2+ beats
    # (IOI then ≈ note+rest). Fall back to onset IOI, then audio beat_track.
    if dur_bpm is not None and (
        ioi_bpm is None
        or abs(np.log2(max(ioi_bpm, 1e-6) / max(dur_bpm, 1e-6))) > 0.4
    ):
        # Duration prior wins when IOI looks octave-ambiguous; lightly
        # pull toward audio only if audio already agrees within ~a fifth.
        if abs(np.log2(max(audio_bpm, 1e-6) / max(dur_bpm, 1e-6))) < 0.3:
            bpm = 0.35 * audio_bpm + 0.65 * dur_bpm
            method = "blend_beat_track_note_duration"
        else:
            bpm = dur_bpm
            method = "note_duration_tempo"
        bpm = _clamp_bpm(bpm, min_bpm=min_bpm, max_bpm=max_bpm)
    elif ioi_bpm is not None:
        ratio = max(bpm, ioi_bpm) / max(1e-6, min(bpm, ioi_bpm))
        if ratio < 1.35 or len(beat_times_list) < 4:
            bpm = (1.0 - IOI_TEMPO_WEIGHT) * audio_bpm + IOI_TEMPO_WEIGHT * ioi_bpm
            method = "blend_beat_track_note_ioi"
        bpm = _clamp_bpm(bpm, min_bpm=min_bpm, max_bpm=max_bpm)
    elif dur_bpm is not None and len(beat_times_list) < 4:
        bpm = 0.35 * audio_bpm + 0.65 * dur_bpm
        method = "blend_beat_track_note_duration"
        bpm = _clamp_bpm(bpm, min_bpm=min_bpm, max_bpm=max_bpm)

    # Resolve half/double tempo using sounding-note duration fit.
    if note_durations:
        resolved, resolve_tag = _pick_bpm_with_octave_fix(
            bpm,
            note_durations,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
        )
        bpm = resolved
        method = f"{method}+{resolve_tag}"

    # Rebuild a uniform beat grid from the chosen BPM anchored at the
    # first detected beat (or first note onset / 0).
    beat_period = 60.0 / bpm
    if beat_times_list:
        anchor = beat_times_list[0]
    elif note_onsets:
        anchor = float(note_onsets[0])
    else:
        anchor = 0.0

    duration_s = float(len(audio) / sr)
    grid: list[float] = []
    t = anchor
    # Walk backward to t≈0 then forward through the clip.
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
        "note_ioi_bpm": round(float(ioi_bpm), 2) if ioi_bpm is not None else None,
        "note_duration_bpm": (
            round(float(dur_bpm), 2) if dur_bpm is not None else None
        ),
        "librosa_beat_times": beat_times_list,
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
    catalog: Sequence[tuple[float, str]] = DURATION_CATALOG,
) -> tuple[str, float]:
    """Map a duration in beats to the nearest named note/rest value."""
    if duration_beats <= 0:
        return "zero", 0.0
    best_name = catalog[-1][1]
    best_beats = catalog[-1][0]
    best_err = abs(duration_beats - best_beats)
    for beats, name in catalog:
        err = abs(duration_beats - beats)
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
) -> QuantizedRhythm:
    """Convert a time interval into beat-relative quantized fields."""
    onset_beats = seconds_to_beats(onset_sec - tempo_anchor_sec, bpm)
    duration_beats = seconds_to_beats(duration_sec, bpm)
    q_onset = quantize_beats(onset_beats, grid=grid)
    # Named catalog snap (quarter/eighth/…) — name and beats always match.
    name, q_dur = nearest_duration_value(max(duration_beats, 0.0))
    if duration_beats > 0 and q_dur <= 0:
        name, q_dur = nearest_duration_value(grid)
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
    include_leading_rests: bool = True,
    include_trailing_rests: bool = True,
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
        )
        # Drop rests that quantize away to nothing / grace-note silence.
        if q.quantized_duration_beats < min_rest_beats and q.duration_beats < min_rest_beats:
            continue
        entry = asdict(r)
        entry.update(_rhythm_fields(q))
        rests_out.append(entry)

    # Chronological mixed timeline for sheet-oriented consumers.
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
        "note_rhythm": note_rhythm,
        "rests": rests_out,
        "num_rests": len(rests_out),
        "events": events,
    }
