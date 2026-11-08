"""Export quantized note/rest timelines to MusicXML (bass clef / cello).

Produces a minimal partwise MusicXML 3.1 score that opens in MuseScore,
Finale, Dorico, etc. No third-party music library required.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

# Quarter note = 1.0 beat. Divisions chosen so 1/32 note = 1 division unit.
DIVISIONS = 8  # 1 beat = 8 divisions; sixteenth = 2; 32nd = 1

# duration_value name → (MusicXML <type> text, dotted?, beats)
_DURATION_TYPES: dict[str, tuple[str, bool, float]] = {
    "whole": ("whole", False, 4.0),
    "dotted_half": ("half", True, 3.0),
    "half": ("half", False, 2.0),
    "dotted_quarter": ("quarter", True, 1.5),
    "quarter": ("quarter", False, 1.0),
    "dotted_eighth": ("eighth", True, 0.75),
    "eighth": ("eighth", False, 0.5),
    "dotted_sixteenth": ("16th", True, 0.375),
    "sixteenth": ("16th", False, 0.25),
    "thirty_second": ("32nd", False, 0.125),
}

# MIDI pitch-class → (step, alter)
_PC_TO_STEP_ALTER: tuple[tuple[str, int], ...] = (
    ("C", 0),
    ("C", 1),  # C#
    ("D", 0),
    ("D", 1),  # D#
    ("E", 0),
    ("F", 0),
    ("F", 1),  # F#
    ("G", 0),
    ("G", 1),  # G#
    ("A", 0),
    ("A", 1),  # A#
    ("B", 0),
)


def midi_to_pitch(midi: int) -> tuple[str, int, int]:
    """Return (step, alter, octave) for a MIDI note number."""
    midi = int(midi)
    step, alter = _PC_TO_STEP_ALTER[midi % 12]
    octave = midi // 12 - 1
    return step, alter, octave


def beats_to_divisions(beats: float, *, divisions: int = DIVISIONS) -> int:
    """Convert beat length to MusicXML duration units (nearest)."""
    return max(1, int(round(float(beats) * divisions)))


def _duration_meta(event: dict) -> tuple[str, bool, float]:
    """Resolve MusicXML type / dot / beats for one timeline event."""
    name = event.get("duration_value")
    if name and name in _DURATION_TYPES:
        return _DURATION_TYPES[name]
    beats = float(
        event.get("duration_value_beats")
        or event.get("quantized_duration_beats")
        or event.get("duration_beats")
        or 1.0
    )
    best = min(
        _DURATION_TYPES.values(),
        key=lambda item: abs(item[2] - beats),
    )
    return best


def _iter_score_events(result: dict) -> list[dict]:
    """Prefer chronological ``events``; else notes (+ rests) by onset."""
    events = result.get("events")
    if events:
        return list(events)
    out: list[dict] = []
    for n in result.get("notes") or []:
        item = dict(n)
        item.setdefault("type", "note")
        out.append(item)
    for r in result.get("rests") or []:
        item = dict(r)
        item.setdefault("type", "rest")
        out.append(item)
    out.sort(
        key=lambda e: (
            float(e.get("onset", 0.0)),
            0 if e.get("type") == "note" else 1,
        )
    )
    return out


def _split_beats_across_measures(
    beats: float,
    *,
    remaining_in_measure: float,
    measure_beats: float,
) -> list[float]:
    """Split a duration into chunks that fit inside successive measures."""
    chunks: list[float] = []
    left = float(beats)
    room = float(remaining_in_measure)
    if room <= 1e-9:
        room = measure_beats
    while left > 1e-9:
        take = min(left, room)
        if take <= 1e-9:
            room = measure_beats
            continue
        chunks.append(round(take, 6))
        left -= take
        room -= take
        if room <= 1e-9:
            room = measure_beats
    return chunks


def _beats_to_type_and_dots(beats: float) -> tuple[str, int]:
    """Map a beat length to MusicXML type + dot count."""
    for _name, (typ, dotted, b) in _DURATION_TYPES.items():
        if abs(b - beats) < 1e-6:
            return typ, (1 if dotted else 0)
    for typ, b in (
        ("whole", 4.0),
        ("half", 2.0),
        ("quarter", 1.0),
        ("eighth", 0.5),
        ("16th", 0.25),
        ("32nd", 0.125),
    ):
        if abs(b - beats) < 1e-6:
            return typ, 0
        if abs(b * 1.5 - beats) < 1e-6:
            return typ, 1
    for typ, b in (
        ("whole", 4.0),
        ("half", 2.0),
        ("quarter", 1.0),
        ("eighth", 0.5),
        ("16th", 0.25),
        ("32nd", 0.125),
    ):
        if b <= beats + 1e-9:
            return typ, 0
    return "32nd", 0


def _append_pitch(note_el: ET.Element, midi: int) -> None:
    step, alter, octave = midi_to_pitch(midi)
    pitch = ET.SubElement(note_el, "pitch")
    ET.SubElement(pitch, "step").text = step
    if alter:
        ET.SubElement(pitch, "alter").text = str(alter)
    ET.SubElement(pitch, "octave").text = str(octave)


def _append_note_element(
    measure: ET.Element,
    *,
    is_rest: bool,
    midi: int | None,
    beats: float,
    divisions: int,
    tie: str | None = None,
) -> None:
    """Append one <note> (sounding or rest) for ``beats`` length."""
    note_el = ET.SubElement(measure, "note")
    if is_rest:
        ET.SubElement(note_el, "rest")
    else:
        assert midi is not None
        _append_pitch(note_el, midi)

    dur_div = beats_to_divisions(beats, divisions=divisions)
    ET.SubElement(note_el, "duration").text = str(dur_div)
    if tie in ("start", "stop", "continue"):
        if tie == "continue":
            ET.SubElement(note_el, "tie", type="stop")
            ET.SubElement(note_el, "tie", type="start")
        else:
            ET.SubElement(note_el, "tie", type=tie)

    ET.SubElement(note_el, "voice").text = "1"
    typ, dots = _beats_to_type_and_dots(beats)
    ET.SubElement(note_el, "type").text = typ
    for _ in range(dots):
        ET.SubElement(note_el, "dot")

    if tie in ("start", "stop", "continue"):
        notations = ET.SubElement(note_el, "notations")
        if tie == "continue":
            ET.SubElement(notations, "tied", type="stop")
            ET.SubElement(notations, "tied", type="start")
        else:
            ET.SubElement(notations, "tied", type=tie)


def _pad_rest_chunks(beats: float) -> list[float]:
    """Greedy binary rest fill for incomplete final measures."""
    chunks: list[float] = []
    left = float(beats)
    for size in (4.0, 2.0, 1.0, 0.5, 0.25, 0.125):
        while left >= size - 1e-9:
            chunks.append(size)
            left -= size
    if left > 1e-6:
        chunks.append(max(round(left, 6), 0.125))
    return chunks


def build_musicxml(
    result: dict,
    *,
    title: str | None = None,
    part_name: str = "Cello",
    measure_beats: float = 4.0,
    divisions: int = DIVISIONS,
) -> ET.Element:
    """Build a partwise MusicXML tree from a CLI result dict.

    Uses quantized ``duration_value`` / beat lengths from the rhythm layer
    and places events sequentially (monophonic). Bass clef, 4/4, C major.
    """
    events = _iter_score_events(result)
    bpm = 120.0
    tempo = result.get("tempo")
    if isinstance(tempo, dict) and tempo.get("bpm"):
        bpm = float(tempo["bpm"])

    if title is None:
        audio_path = result.get("audio_path")
        title = Path(audio_path).stem if audio_path else "Cello transcription"

    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title

    identification = ET.SubElement(score, "identification")
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "cello-note-detector"

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = part_name

    part = ET.SubElement(score, "part", id="P1")

    measure_num = 1
    measure = ET.SubElement(part, "measure", number=str(measure_num))
    attrs = ET.SubElement(measure, "attributes")
    ET.SubElement(attrs, "divisions").text = str(divisions)
    key = ET.SubElement(attrs, "key")
    ET.SubElement(key, "fifths").text = "0"
    time_el = ET.SubElement(attrs, "time")
    ET.SubElement(time_el, "beats").text = str(int(measure_beats))
    ET.SubElement(time_el, "beat-type").text = "4"
    clef = ET.SubElement(attrs, "clef")
    ET.SubElement(clef, "sign").text = "F"
    ET.SubElement(clef, "line").text = "4"

    direction = ET.SubElement(measure, "direction", placement="above")
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = f"{bpm:.0f}"
    ET.SubElement(direction, "sound", tempo=f"{bpm:.2f}")

    filled = 0.0

    if not events:
        _append_note_element(
            measure,
            is_rest=True,
            midi=None,
            beats=measure_beats,
            divisions=divisions,
        )
        return score

    for event in events:
        is_rest = event.get("type") == "rest" or (
            "midi" not in event and event.get("kind") == "rest"
        )
        _, _, catalog_beats = _duration_meta(event)
        beats = float(
            event.get("duration_value_beats")
            or event.get("quantized_duration_beats")
            or catalog_beats
        )
        if beats <= 0:
            continue
        midi = None if is_rest else int(event.get("midi", 48))

        remaining = measure_beats - filled
        if remaining <= 1e-9:
            measure_num += 1
            measure = ET.SubElement(part, "measure", number=str(measure_num))
            filled = 0.0
            remaining = measure_beats

        chunks = _split_beats_across_measures(
            beats,
            remaining_in_measure=remaining,
            measure_beats=measure_beats,
        )

        n_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            if filled >= measure_beats - 1e-9:
                measure_num += 1
                measure = ET.SubElement(
                    part, "measure", number=str(measure_num)
                )
                filled = 0.0

            tie: str | None = None
            if not is_rest and n_chunks > 1:
                if i == 0:
                    tie = "start"
                elif i == n_chunks - 1:
                    tie = "stop"
                else:
                    tie = "continue"

            _append_note_element(
                measure,
                is_rest=is_rest,
                midi=midi,
                beats=chunk,
                divisions=divisions,
                tie=tie,
            )
            filled += chunk

    pad = measure_beats - filled
    if pad > 1e-6:
        for rest_beats in _pad_rest_chunks(pad):
            _append_note_element(
                measure,
                is_rest=True,
                midi=None,
                beats=rest_beats,
                divisions=divisions,
            )

    return score


def musicxml_to_string(root: ET.Element) -> str:
    """Pretty-print MusicXML with XML declaration."""
    rough = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def write_musicxml(
    result: dict,
    path: str | Path,
    *,
    title: str | None = None,
    part_name: str = "Cello",
) -> Path:
    """Write a MusicXML file for ``result``; return the output path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = build_musicxml(result, title=title, part_name=part_name)
    out.write_text(musicxml_to_string(root), encoding="utf-8")
    return out
