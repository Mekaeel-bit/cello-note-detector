#!/usr/bin/env python3
"""Build a playable monophonic catalog score JSON from Mutopia LilyPond.

The eval reference (`prelude_monophonic_reference.json`) has a correct verified
prefix but a broken relative-octave tail (MIDI down to 19). This script
re-parses ``mutopia517_prelude.ly`` with LilyPond-like diatonic relative rules
and writes ``prelude_catalog_score.json`` for MusicXML/MIDI attach.

Usage:
  python scripts/build_bwv1007_catalog_score.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LY = _REPO / "data" / "reference" / "bwv1007_prelude" / "mutopia517_prelude.ly"
_OUT = _REPO / "data" / "reference" / "bwv1007_prelude" / "prelude_catalog_score.json"

_DIATONIC = {"c": 0, "d": 1, "e": 2, "f": 3, "g": 4, "a": 5, "b": 6}
_STEP = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_DUR_NAME = {
    1: "whole",
    2: "half",
    4: "quarter",
    8: "eighth",
    16: "sixteenth",
    32: "thirty_second",
}
_DOTTED = {
    1: "dotted_half",
    2: "dotted_half",
    4: "dotted_quarter",
    8: "dotted_eighth",
    16: "dotted_sixteenth",
}


def _expand_polyphony(body: str) -> str:
    """Replace << upper \\\\ lower >> with the lower voice (Prelude broken chords)."""
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
    """Collapse <pitches>dur to the lowest written pitch + duration."""

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
    """Choose octave by diatonic staff proximity (LilyPond-like), then apply marks."""
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
        # Minimize absolute staff distance; ties prefer the higher pitch.
        key = (abs(dist), -dist)
        if best is None or key < best_key:
            best_key = key
            best = midi
    assert best is not None
    for ch in oct_marks:
        best += 12 if ch == "'" else -12
    return best


def parse_prelude_ly(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\\relative\s+c'\s*\{([\s\S]*)\}\s*$", text)
    if not match:
        raise SystemExit(f"Could not find \\relative c' block in {path}")
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
    body = body.replace("(", " ").replace(")", " ").replace("!", "")
    body = re.sub(r"\s+", " ", body)

    note_re = re.compile(r"([a-g](?:is|es)?)([,']*)(16|32|64|8|4|2|1)?(\.*)(~)?")
    prev = 60  # written c'
    prev_name = "c"
    cur_dur = 16
    beat = 0.0
    events: list[dict] = []
    for m in note_re.finditer(body):
        name, marks, dur, dots, _tie = m.groups()
        written = _apply_relative(prev, prev_name, name, marks)
        prev = written
        prev_name = name
        if dur:
            cur_dur = int(dur)
        beats = 4.0 / cur_dur
        if dots:
            beats *= 2 - 0.5 ** len(dots)
            dur_value = _DOTTED.get(cur_dur, _DUR_NAME[cur_dur])
        else:
            dur_value = _DUR_NAME[cur_dur]
        sounding = written - 12  # Mutopia transpose c → c,
        # Final open-G safety: never emit below cello C2 for this catalog score.
        if sounding < 36:
            sounding += 12
        events.append(
            {
                "midi": int(sounding),
                "onset_beats": round(beat, 6),
                "duration_beats": float(beats),
                "duration_value": dur_value,
            }
        )
        beat += beats
    return events


def main() -> int:
    events = parse_prelude_ly(_LY)
    midi_sequence = [int(e["midi"]) for e in events]
    payload = {
        "title": "Bach Cello Suite No. 1, BWV 1007 — Prélude",
        "source_edition": "Mutopia-2018/01/19-517 after Schirmer, 1916 (Public Domain)",
        "imslp_work": (
            "https://imslp.org/wiki/Cello_Suite_No.1_in_G_major,"
            "_BWV_1007_(Bach,_Johann_Sebastian)"
        ),
        "notes": (
            "Playable monophonic catalog score regenerated from mutopia517_prelude.ly "
            "with LilyPond-like diatonic relative octave rules + c→c, transpose. "
            "Polyphonic <<>> bars use the lower voice. Use this file for MusicXML/MIDI "
            "attach. The eval reference JSON keeps a hand-checked prefix for metrics."
        ),
        "key": "G major",
        "time_signature": "4/4",
        "typical_unit": "sixteenth",
        "num_notes": len(midi_sequence),
        "total_beats": float(events[-1]["onset_beats"] + events[-1]["duration_beats"]),
        "midi_sequence": midi_sequence,
        "events": events,
    }
    _OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {_OUT.relative_to(_REPO)} notes={payload['num_notes']} "
        f"beats={payload['total_beats']} "
        f"midi={min(midi_sequence)}-{max(midi_sequence)}"
    )

    # Sanity vs verified eval prefix.
    eval_path = _REPO / "data/reference/bwv1007_prelude/prelude_monophonic_reference.json"
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    n = int(eval_data.get("verified_prefix_notes", 128))
    if midi_sequence[:n] != eval_data["midi_sequence"][:n]:
        print(f"WARNING: first {n} notes differ from eval reference", file=sys.stderr)
        return 1
    print(f"Verified prefix-{n} matches eval reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
