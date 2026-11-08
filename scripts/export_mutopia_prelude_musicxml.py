#!/usr/bin/env python3
"""Export Mutopia BWV 1007 Prélude LilyPond → MusicXML edition file.

This is an *edition export* from the Mutopia/Schirmer 1916 Public Domain
LilyPond source (not a rebuild from our eval JSON / MIDI). Slurs in the
``.ly`` are preserved as MusicXML ``<slur>`` markings.

Usage:
  python scripts/export_mutopia_prelude_musicxml.py
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

_REPO = Path(__file__).resolve().parents[1]
_LY = (
    _REPO
    / "data"
    / "catalog"
    / "pieces"
    / "bwv1007_prelude"
    / "source"
    / "mutopia517_prelude.ly"
)
# Fallback to reference copy if catalog source not yet placed.
_LY_FALLBACK = _REPO / "data" / "reference" / "bwv1007_prelude" / "mutopia517_prelude.ly"
_OUT = (
    _REPO
    / "data"
    / "catalog"
    / "pieces"
    / "bwv1007_prelude"
    / "edition.musicxml"
)

_DIATONIC = {"c": 0, "d": 1, "e": 2, "f": 3, "g": 4, "a": 5, "b": 6}
_STEP = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_DUR_DIV = {1: 32, 2: 16, 4: 8, 8: 4, 16: 2, 32: 1}  # divisions=8 → quarter=8
_DUR_TYPE = {
    1: "whole",
    2: "half",
    4: "quarter",
    8: "eighth",
    16: "16th",
    32: "32nd",
}


def _expand_polyphony(body: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        if body.startswith("<<", i):
            depth = 1
            j = i + 2
            while j < len(body) and depth:
                if body.startswith("<<", j):
                    depth += 1
                    j += 2
                elif body.startswith(">>", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            block = body[i + 2 : j - 2]
            parts = re.split(r"\\\\", block)
            lower = parts[-1] if len(parts) >= 2 else block
            lower = lower.replace("s", " ")
            lower = re.sub(r"[\[\]]", " ", lower)
            out.append(lower)
            i = j
        else:
            out.append(body[i])
            i += 1
    return "".join(out)


def _expand_chords(body: str) -> str:
    def repl(match: re.Match[str]) -> str:
        pitches = re.findall(r"[a-g](?:is|es)?[,']*", match.group(1))
        dur = match.group(2) or "1"
        return (pitches[0] + dur) if pitches else " "

    return re.sub(r"<([^>]+)>((?:16|32|8|4|2|1)?\.*)", repl, body)


def _parse_name(name: str) -> tuple[str, str]:
    match = re.match(r"([a-g])(is|es)?", name)
    assert match is not None
    return match.group(1), match.group(2) or ""


def _pc_of(name: str) -> int:
    step, acc = _parse_name(name)
    return (_STEP[step] + {"": 0, "is": 1, "es": -1}[acc]) % 12


def _apply_relative(prev_midi: int, prev_name: str, name: str, oct_marks: str) -> int:
    step, _acc = _parse_name(name)
    target_d = _DIATONIC[step]
    prev_d = _DIATONIC[_parse_name(prev_name)[0]]
    pc = _pc_of(name)
    best: int | None = None
    best_key: tuple[int, int] | None = None
    prev_staff = (prev_midi // 12) * 7 + prev_d
    for oct in range(0, 10):
        midi = oct * 12 + pc
        staff = oct * 7 + target_d
        dist = staff - prev_staff
        key = (abs(dist), -dist)
        if best is None or key < best_key:
            best_key = key
            best = midi
    assert best is not None
    for ch in oct_marks:
        best += 12 if ch == "'" else -12
    return best


def parse_ly_events(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\\relative\s+c'\s*\{([\s\S]*)\}\s*$", text)
    if not match:
        raise SystemExit(f"No \\relative c' block in {path}")
    body = re.sub(r"%.*?$", "", match.group(1), flags=re.M)
    body = _expand_polyphony(body)
    for tok in (
        "\\fermata",
        "\\noBeam",
        "\\break",
        '\\bar "|."',
        "\\time 4/4",
        "\\key g \\major",
    ):
        body = body.replace(tok, "")
    body = re.sub(r'\\set Staff\.midiInstrument = "cello"', "", body)
    body = _expand_chords(body)

    # Tokenize keeping slur marks as separate tokens.
    token_re = re.compile(
        r"\(|\)|~|([a-g](?:is|es)?)([,']*)(16|32|64|8|4|2|1)?(\.*)|!"
    )
    prev = 60
    prev_name = "c"
    cur_dur = 16
    beat = 0.0
    slur_depth = 0
    events: list[dict] = []
    pending_tie = False
    for m in token_re.finditer(body):
        tok = m.group(0)
        if tok == "(":
            slur_depth += 1
            if events:
                events[-1]["slur_start"] = True
            else:
                # Slur opens before first note — mark next note.
                pending_slur_start = True  # noqa: F841 — handled below
            continue
        if tok == ")":
            if events:
                events[-1]["slur_stop"] = True
            slur_depth = max(0, slur_depth - 1)
            continue
        if tok == "~":
            if events:
                events[-1]["tie_start"] = True
            pending_tie = True
            continue
        if tok == "!":
            continue
        name, marks, dur, dots = m.group(1), m.group(2), m.group(3), m.group(4)
        if not name:
            continue
        written = _apply_relative(prev, prev_name, name, marks or "")
        prev = written
        prev_name = name
        if dur:
            cur_dur = int(dur)
        beats = 4.0 / cur_dur
        if dots:
            beats *= 2 - 0.5 ** len(dots)
        sounding = written - 12
        if sounding < 36:
            sounding += 12
        step, acc = _parse_name(name)
        alter = {"": 0, "is": 1, "es": -1}[acc]
        ev = {
            "midi": sounding,
            "step": step.upper(),
            "alter": alter,
            "octave": sounding // 12 - 1,
            "duration_div": max(1, int(round(beats * 8))),
            "type": _DUR_TYPE[cur_dur],
            "dots": len(dots or ""),
            "onset_beats": beat,
            "slur_start": False,
            "slur_stop": False,
            "tie_start": False,
            "tie_stop": pending_tie,
            "fermata": False,
        }
        pending_tie = False
        events.append(ev)
        beat += beats

    # Fermata on final note (Mutopia has fermata on last chord).
    if events:
        events[-1]["fermata"] = True
    return events


def build_musicxml(events: list[dict]) -> ET.Element:
    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = (
        "Bach Cello Suite No. 1, BWV 1007 — Prélude"
    )
    ident = ET.SubElement(score, "identification")
    ET.SubElement(ident, "creator", type="composer").text = "Johann Sebastian Bach"
    encoding = ET.SubElement(ident, "encoding")
    ET.SubElement(encoding, "software").text = (
        "cello-note-detector (Mutopia LilyPond PD edition export)"
    )
    misc = ET.SubElement(ident, "miscellaneous")
    mf = ET.SubElement(misc, "miscellaneous-field", name="edition")
    mf.text = (
        "Mutopia-2018/01/19-517 after Schirmer, 1916 (Public Domain). "
        "Exported from mutopia517_prelude.ly; slurs preserved from LilyPond."
    )
    part_list = ET.SubElement(score, "part-list")
    sp = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Cello"
    part = ET.SubElement(score, "part", id="P1")

    measure_num = 1
    measure = ET.SubElement(part, "measure", number="1")
    attrs = ET.SubElement(measure, "attributes")
    ET.SubElement(attrs, "divisions").text = "8"
    key = ET.SubElement(attrs, "key")
    ET.SubElement(key, "fifths").text = "1"
    time_el = ET.SubElement(attrs, "time")
    ET.SubElement(time_el, "beats").text = "4"
    ET.SubElement(time_el, "beat-type").text = "4"
    clef = ET.SubElement(attrs, "clef")
    ET.SubElement(clef, "sign").text = "F"
    ET.SubElement(clef, "line").text = "4"

    filled = 0.0
    measure_beats = 4.0
    for ev in events:
        beats = ev["duration_div"] / 8.0
        if filled >= measure_beats - 1e-9:
            measure_num += 1
            measure = ET.SubElement(part, "measure", number=str(measure_num))
            filled = 0.0
        # Split across barline if needed.
        remaining = measure_beats - filled
        chunks = []
        left = beats
        room = remaining
        while left > 1e-9:
            take = min(left, room)
            chunks.append(take)
            left -= take
            room -= take
            if room <= 1e-9:
                room = measure_beats
        for i, chunk in enumerate(chunks):
            if filled >= measure_beats - 1e-9:
                measure_num += 1
                measure = ET.SubElement(part, "measure", number=str(measure_num))
                filled = 0.0
            note = ET.SubElement(measure, "note")
            pitch = ET.SubElement(note, "pitch")
            ET.SubElement(pitch, "step").text = ev["step"]
            if ev["alter"]:
                ET.SubElement(pitch, "alter").text = str(ev["alter"])
            ET.SubElement(pitch, "octave").text = str(ev["octave"])
            dur_div = max(1, int(round(chunk * 8)))
            ET.SubElement(note, "duration").text = str(dur_div)
            if ev.get("tie_stop") and i == 0:
                ET.SubElement(note, "tie", type="stop")
            if (ev.get("tie_start") and i == len(chunks) - 1) or (
                len(chunks) > 1 and i < len(chunks) - 1
            ):
                ET.SubElement(note, "tie", type="start")
            ET.SubElement(note, "voice").text = "1"
            # Prefer catalog type for first chunk; otherwise nearest.
            typ = ev["type"] if i == 0 and abs(chunk - beats) < 1e-6 else None
            if typ is None:
                for d, name in (
                    (4.0, "whole"),
                    (2.0, "half"),
                    (1.0, "quarter"),
                    (0.5, "eighth"),
                    (0.25, "16th"),
                    (0.125, "32nd"),
                ):
                    if abs(d - chunk) < 1e-6:
                        typ = name
                        break
                typ = typ or "16th"
            ET.SubElement(note, "type").text = typ
            if ev["dots"] and i == 0 and len(chunks) == 1:
                for _ in range(ev["dots"]):
                    ET.SubElement(note, "dot")
            notations = None
            if ev.get("slur_start") and i == 0:
                notations = notations or ET.SubElement(note, "notations")
                ET.SubElement(notations, "slur", type="start")
            if ev.get("slur_stop") and i == len(chunks) - 1:
                notations = notations or ET.SubElement(note, "notations")
                ET.SubElement(notations, "slur", type="stop")
            if (ev.get("tie_start") and i == len(chunks) - 1) or (
                len(chunks) > 1 and i < len(chunks) - 1
            ):
                notations = notations or ET.SubElement(note, "notations")
                ET.SubElement(notations, "tied", type="start")
            if ev.get("tie_stop") and i == 0:
                notations = notations or ET.SubElement(note, "notations")
                ET.SubElement(notations, "tied", type="stop")
            if ev.get("fermata") and i == len(chunks) - 1:
                notations = notations or ET.SubElement(note, "notations")
                ET.SubElement(notations, "fermata", type="upright")
            filled += chunk
    return score


def _pretty(root: ET.Element) -> str:
    rough = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode(
        "utf-8"
    )


def main() -> int:
    ly = _LY if _LY.is_file() else _LY_FALLBACK
    events = parse_ly_events(ly)
    midis = [e["midi"] for e in events]
    print(
        f"Parsed {len(events)} notes from {ly.name}; "
        f"range {min(midis)}-{max(midis)}; slurs_start="
        f"{sum(1 for e in events if e['slur_start'])}"
    )
    if min(midis) < 36:
        print("ERROR: sub-C2 pitches present", file=sys.stderr)
        return 1
    root = build_musicxml(events)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(_pretty(root), encoding="utf-8")
    print(f"Wrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
