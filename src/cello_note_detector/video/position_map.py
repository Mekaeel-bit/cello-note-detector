"""Map fused left-hand position / string → engraved score labels.

Sticky ``lh_position`` from fusion is preferred. Fallback from
``semitones_above_open``:

| semitones | score label |
|----------:|-------------|
| 0–4       | I           |
| 5–7       | II          |
| 8–10      | III         |
| 11–14     | IV          |
| ≥15       | high / thumb (thumb on A/D) |

MusicXML ``<string>`` numbers: **1 = highest** → A=1, D=2, G=3, C=4.

Fingering on the score comes from fusion ``finger = f(position, string, pitch)``
(open = 0 only when sounding the open string).
"""

from __future__ import annotations

from typing import Any

CELLO_STRING_TO_NUMBER: dict[str, int] = {"A": 1, "D": 2, "G": 3, "C": 4}
CELLO_NUMBER_TO_STRING: dict[int, str] = {v: k for k, v in CELLO_STRING_TO_NUMBER.items()}

POSITION_MAPPING_DOC = (
    "sticky lh_position from neck+pitch; else semitones_above_open → "
    "I (0–4), II (5–7), III (8–10), IV (11–14), high/thumb (≥15; thumb on A/D). "
    "Finger = f(position, string, pitch); 0 only on open string."
)


def position_label(
    *,
    semitones_above_open: int | None = None,
    position_band: str | None = None,
    string_name: str | None = None,
    lh_position: str | None = None,
) -> str:
    """Return engraved left-hand position text (I–IV / thumb / high)."""
    if lh_position:
        return str(lh_position)
    if semitones_above_open is not None:
        st = int(semitones_above_open)
        if st <= 4:
            return "I"
        if st <= 7:
            return "II"
        if st <= 10:
            return "III"
        if st <= 14:
            return "IV"
        if (string_name or "").upper() in {"A", "D"}:
            return "thumb"
        return "high"

    band = (position_band or "").lower()
    if band in {"open", "low"}:
        return "I"
    if band == "mid":
        return "II"
    if band == "high":
        if (string_name or "").upper() in {"A", "D"}:
            return "thumb"
        return "high"
    return "I"


def musicxml_string_number(string_name: str | None) -> int | None:
    if not string_name:
        return None
    return CELLO_STRING_TO_NUMBER.get(string_name.upper())


def annotate_fused_note(note: dict[str, Any]) -> dict[str, Any]:
    """Attach score-oriented fingering / position / string fields onto a fused note."""
    fing = note.get("fingering") or {}
    finger = fing.get("finger")
    if finger is None and "suggested_finger" in note:
        finger = note.get("suggested_finger")
    string_name = note.get("string") or note.get("bow_string")
    st = note.get("semitones_above_open")
    if st is None and note.get("hypotheses"):
        st = note["hypotheses"][0].get("semitones_above_open")
    lh = note.get("lh_position") or fing.get("lh_position")
    label = position_label(
        semitones_above_open=int(st) if st is not None else None,
        position_band=note.get("position_band"),
        string_name=string_name,
        lh_position=lh,
    )
    out = dict(note)
    out["score_finger"] = int(finger) if finger is not None else None
    out["score_position"] = label
    out["score_string"] = string_name
    out["score_string_number"] = musicxml_string_number(string_name)
    return out
