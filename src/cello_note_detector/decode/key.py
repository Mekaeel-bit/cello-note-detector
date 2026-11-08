"""Key-signature inference from detected pitches (not filenames).

Uses a duration-weighted pitch-class histogram scored against the
Krumhansl–Schmuckler major/minor profiles, with a small mode tie-break
that prefers accidentals actually present in the clip (e.g. ♭3 / ♭7).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cello_note_detector.decode.events import NoteEvent

# Krumhansl–Kessler key profiles (tonic at index 0).
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

_PC_NAMES = (
    "C",
    "C#",
    "D",
    "Eb",
    "E",
    "F",
    "F#",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
)

# Circle-of-fifths offset from C for major keys; minor uses relative major.
_MAJOR_FIFTHS = {
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
    "F#": 6,
    "C#": 7,
    "F": -1,
    "Bb": -2,
    "Eb": -3,
    "Ab": -4,
    "Db": -5,
    "Gb": -6,
    "Cb": -7,
}

# Acceptable CLI / override spellings → (tonic_pc, mode, display_name, fifths)
_KEY_ALIASES: dict[str, tuple[int, str, str, int]] = {}


def _register_key(name: str, tonic_pc: int, mode: str, fifths: int) -> None:
    display = f"{_PC_NAMES[tonic_pc]} {mode}"
    _KEY_ALIASES[name.lower()] = (tonic_pc, mode, display, fifths)


for _tonic, _fifths in _MAJOR_FIFTHS.items():
    _pc = _PC_NAMES.index(_tonic) if _tonic in _PC_NAMES else None
    if _pc is None:
        continue
    _register_key(_tonic, _pc, "major", _fifths)
    _register_key(f"{_tonic} major", _pc, "major", _fifths)
    _register_key(f"{_tonic}maj", _pc, "major", _fifths)
    # Relative minor: tonic = major tonic - 3 semitones; same fifths.
    _min_pc = (_pc - 3) % 12
    _min_name = _PC_NAMES[_min_pc]
    _register_key(f"{_min_name}m", _min_pc, "minor", _fifths)
    _register_key(f"{_min_name} minor", _min_pc, "minor", _fifths)
    _register_key(f"{_min_name}min", _min_pc, "minor", _fifths)

# Extra enharmonic / common aliases.
_register_key("bb", 10, "major", -2)
_register_key("bb major", 10, "major", -2)
_register_key("bbm", 10, "minor", -5)
_register_key("bb minor", 10, "minor", -5)
_register_key("eb", 3, "major", -3)
_register_key("eb major", 3, "major", -3)
_register_key("ebm", 3, "minor", -6)
_register_key("eb minor", 3, "minor", -6)
_register_key("ab", 8, "major", -4)
_register_key("ab major", 8, "major", -4)
_register_key("db", 1, "major", -5)
_register_key("gb", 6, "major", -6)


def parse_key(spec: str) -> dict:
    """Parse a CLI key string like ``Gm``, ``G minor``, or ``Bb``."""
    key = spec.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    if key not in _KEY_ALIASES:
        raise ValueError(
            f"Unrecognized key '{spec}'. Try e.g. C, Gm, Bb, D major, A minor."
        )
    tonic_pc, mode, name, fifths = _KEY_ALIASES[key]
    return {
        "name": name,
        "tonic": _PC_NAMES[tonic_pc],
        "tonic_pc": tonic_pc,
        "mode": mode,
        "fifths": fifths,
        "method": "cli_override",
        "confidence": 1.0,
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def pitch_class_histogram(
    notes: Sequence[NoteEvent] | Sequence[dict],
    *,
    min_duration: float = 0.12,
    min_confidence: float = 0.25,
) -> np.ndarray:
    """Duration-weighted pitch-class histogram (ignores short flicker)."""
    hist = np.zeros(12, dtype=np.float64)
    for n in notes:
        if isinstance(n, dict):
            midi = int(n["midi"])
            dur = float(n.get("duration", 0.0))
            conf = float(n.get("confidence", 1.0))
        else:
            midi = int(n.midi)
            dur = float(n.duration)
            conf = float(n.confidence)
        if dur < min_duration or conf < min_confidence:
            continue
        hist[midi % 12] += dur
    return hist


def _mode_tiebreak_bonus(hist: np.ndarray, tonic_pc: int, mode: str) -> float:
    """Small bonus when chromatics match major vs minor for the same tonic."""
    # Scale degrees relative to tonic.
    deg = {i: float(hist[(tonic_pc + i) % 12]) for i in range(12)}
    if mode == "minor":
        # Prefer ♭3 / ♭7 over natural 3 / 7.
        return 0.04 * (
            (1.0 if deg[3] > deg[4] else -1.0)
            + (1.0 if deg[10] >= deg[11] else -1.0)
        )
    # Major: prefer natural 3 / 7.
    return 0.04 * (
        (1.0 if deg[4] > deg[3] else -1.0)
        + (1.0 if deg[11] >= deg[10] else -1.0)
    )


def _fifths_for(tonic_pc: int, mode: str) -> int:
    tonic = _PC_NAMES[tonic_pc]
    if mode == "major":
        return int(_MAJOR_FIFTHS.get(tonic, 0))
    # Minor shares the relative major's key signature.
    rel_major_pc = (tonic_pc + 3) % 12
    return int(_MAJOR_FIFTHS.get(_PC_NAMES[rel_major_pc], 0))


def infer_key(
    notes: Sequence[NoteEvent] | Sequence[dict],
    *,
    min_duration: float = 0.12,
    min_confidence: float = 0.25,
) -> dict:
    """Infer key signature from detected note pitches.

    Returns a JSON-ready dict with ``name``, ``tonic``, ``mode``, ``fifths``,
    ``confidence``, and ``method``. Defaults to C major when evidence is thin.
    """
    hist = pitch_class_histogram(
        notes, min_duration=min_duration, min_confidence=min_confidence
    )
    total = float(hist.sum())
    if total < 0.25:
        return {
            "name": "C major",
            "tonic": "C",
            "tonic_pc": 0,
            "mode": "major",
            "fifths": 0,
            "method": "default_insufficient_pitch",
            "confidence": 0.0,
            "pitch_class_weights": [round(float(x), 4) for x in hist.tolist()],
        }

    scored: list[tuple[float, int, str]] = []
    for tonic_pc in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(profile, tonic_pc)
            score = _corr(hist, rotated) + _mode_tiebreak_bonus(
                hist, tonic_pc, mode
            )
            scored.append((score, tonic_pc, mode))
    scored.sort(reverse=True)
    best_score, tonic_pc, mode = scored[0]
    second = scored[1][0] if len(scored) > 1 else -1.0
    confidence = float(np.clip((best_score - max(second, 0.0)) + 0.5 * best_score, 0.0, 1.0))

    return {
        "name": f"{_PC_NAMES[tonic_pc]} {mode}",
        "tonic": _PC_NAMES[tonic_pc],
        "tonic_pc": tonic_pc,
        "mode": mode,
        "fifths": _fifths_for(tonic_pc, mode),
        "method": "krumhansl_schmuckler",
        "confidence": round(confidence, 4),
        "score": round(float(best_score), 4),
        "pitch_class_weights": [round(float(x), 4) for x in hist.tolist()],
    }
