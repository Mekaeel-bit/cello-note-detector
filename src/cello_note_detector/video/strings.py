"""Cello open-string / position hypotheses from sounding MIDI (instrument prior)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Standard cello tuning (MIDI). Instrument prior — not piece-specific.
CELLO_STRINGS: tuple[tuple[str, int], ...] = (
    ("C", 36),  # C2
    ("G", 43),  # G2
    ("D", 50),  # D3
    ("A", 57),  # A3
)
OPEN_STRING_MIDI = {midi: name for name, midi in CELLO_STRINGS}
STRING_ORDER = [name for name, _ in CELLO_STRINGS]  # low → high
OPEN_MIDI_BY_STRING = {name: midi for name, midi in CELLO_STRINGS}

# Left-hand position → approx. semitone of 1st-finger stop above the open string.
# General cello prior (not scale-specific).
POSITION_BASE_ST: dict[str, int] = {
    "I": 0,
    "II": 3,
    "III": 5,
    "IV": 7,
    "thumb": 12,
    "high": 12,
}
POSITION_SEQUENCE: tuple[str, ...] = ("I", "II", "III", "IV", "thumb")


@dataclass(frozen=True)
class StringHypothesis:
    """One way to produce a sounding pitch on the cello."""

    string: str
    open_midi: int
    midi: int
    semitones_above_open: int
    suggested_finger: int
    position_band: str  # "open" | "low" | "mid" | "high"
    rank_score: float  # lower = preferred (prefer lower positions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _position_band(semitones: int) -> str:
    if semitones <= 0:
        return "open"
    if semitones <= 4:
        return "low"
    if semitones <= 8:
        return "mid"
    return "high"


def finger_for_stop(
    semitones_above_open: int,
    *,
    position: str = "I",
) -> int | None:
    """Map a stop on a string in a LH position → finger 0–4.

    Finger **0** only when the sounding pitch is the open string.
    Otherwise chromatic spacing relative to the position's 1st-finger base:
    1st ≈ 1–2 st above base, 2nd ≈ 3–4, 3rd ≈ 5–6, 4th ≈ 7–8.
    Returns ``None`` if the stop is out of comfortable reach (needs a shift).
    """
    st = int(semitones_above_open)
    if st < 0:
        return None
    if st == 0:
        return 0
    base = int(POSITION_BASE_ST.get(position, 0))
    # Open-string pitch while hand is high on the neck is not "finger 0".
    d = st - base
    if d < 1:
        # Below 1st-finger seat in this position — treat as 1 (stopped).
        return 1
    if d <= 2:
        return 1
    if d <= 4:
        return 2
    if d <= 6:
        return 3
    if d <= 8:
        return 4
    return None


def position_needed_for_stop(semitones_above_open: int) -> str:
    """Lowest LH position that can reach ``semitones_above_open`` with finger ≤4."""
    st = int(semitones_above_open)
    for pos in POSITION_SEQUENCE:
        if finger_for_stop(st, position=pos) is not None:
            return pos
    return "high"


def pitch_string_hypotheses(
    midi: int,
    *,
    max_semitones: int = 24,
) -> list[StringHypothesis]:
    """Return feasible (string, stop) hypotheses for a sounding MIDI pitch.

    Ranked by ascending semitones-above-open (prefer lower hand), then lower
    string when tied — a general monophonic prior, not piece-specific.
    ``suggested_finger`` is the **first-position** finger for that stop; callers
    that track sticky LH position should recompute with ``finger_for_stop``.
    """
    midi = int(midi)
    hyps: list[StringHypothesis] = []
    for name, open_midi in CELLO_STRINGS:
        st = midi - open_midi
        if st < 0 or st > max_semitones:
            continue
        finger = finger_for_stop(st, position="I")
        if finger is None:
            finger = 4
        band = _position_band(st)
        string_idx = STRING_ORDER.index(name)
        rank = float(st) + 0.05 * string_idx
        hyps.append(
            StringHypothesis(
                string=name,
                open_midi=open_midi,
                midi=midi,
                semitones_above_open=st,
                suggested_finger=int(finger),
                position_band=band,
                rank_score=round(rank, 4),
            )
        )
    hyps.sort(key=lambda h: (h.rank_score, h.open_midi))
    return hyps


def primary_hypothesis(midi: int, **kwargs) -> StringHypothesis | None:
    hyps = pitch_string_hypotheses(midi, **kwargs)
    return hyps[0] if hyps else None
