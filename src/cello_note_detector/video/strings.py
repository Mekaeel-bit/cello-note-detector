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

# Rough first-position chromatic finger map (semitones above open → finger).
# Open = 0; beyond ~7–8 st is treated as a higher position (thumb/shift proxy).
_FIRST_POS_FINGER = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    7: 4,
    8: 4,
}


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


def _suggested_finger(semitones: int) -> int:
    if semitones < 0:
        return -1
    if semitones in _FIRST_POS_FINGER:
        return _FIRST_POS_FINGER[semitones]
    # Higher neck: keep finger 4 as a crude stop proxy.
    return 4


def pitch_string_hypotheses(
    midi: int,
    *,
    max_semitones: int = 24,
) -> list[StringHypothesis]:
    """Return feasible (string, position) hypotheses for a sounding MIDI pitch.

    A pitch is playable on any string whose open MIDI is ≤ the sounding pitch
    and within ``max_semitones`` above the open string. Hypotheses are ranked
    by ascending position (prefer lower hand / fewer semitones), then by
    lower string when tied — a general monophonic prior, not Bach-specific.
    """
    midi = int(midi)
    hyps: list[StringHypothesis] = []
    for name, open_midi in CELLO_STRINGS:
        st = midi - open_midi
        if st < 0 or st > max_semitones:
            continue
        finger = _suggested_finger(st)
        band = _position_band(st)
        # Prefer lower positions; tiny preference for lower strings at same st
        # (open C pedal vs same pitch high on G is uncommon in first position).
        string_idx = STRING_ORDER.index(name)
        rank = float(st) + 0.05 * string_idx
        hyps.append(
            StringHypothesis(
                string=name,
                open_midi=open_midi,
                midi=midi,
                semitones_above_open=st,
                suggested_finger=finger,
                position_band=band,
                rank_score=round(rank, 4),
            )
        )
    hyps.sort(key=lambda h: (h.rank_score, h.open_midi))
    return hyps


def primary_hypothesis(midi: int, **kwargs) -> StringHypothesis | None:
    hyps = pitch_string_hypotheses(midi, **kwargs)
    return hyps[0] if hyps else None
