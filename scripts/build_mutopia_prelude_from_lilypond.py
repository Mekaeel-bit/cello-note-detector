#!/usr/bin/env python3
"""Rebuild Bach Prélude edition.musicxml from Mutopia LilyPond via LilyPond MIDI.

Requires ``lilypond`` on PATH (tested with 2.24.x). This is the sandbox path that
matches Mutopia pitch/rhythm/tempo ground truth (♩=80, 16ths, peak G4).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LY_DIR = _REPO / "data" / "catalog" / "pieces" / "bwv1007_prelude" / "source"
_OUT = _REPO / "data" / "catalog" / "pieces" / "bwv1007_prelude" / "edition.musicxml"


def main() -> int:
    if subprocess.call(["lilypond", "--version"], stdout=subprocess.DEVNULL) != 0:
        print("lilypond not found on PATH", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name in ("mutopia517_prelude.ly", "header.ily", "bwv1007_prelude.ly"):
            src = _LY_DIR / name
            if src.is_file():
                (tmp_path / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        score = tmp_path / "prelude_score.ly"
        score.write_text(
            r"""\version "2.18.2"
\include "mutopia517_prelude.ly"
\score {
  { \clef "bass" \transpose c c, { \prelude } }
  \layout { }
  \midi { \tempo 4 = 80 }
}
""",
            encoding="utf-8",
        )
        subprocess.check_call(["lilypond", "-o", str(tmp_path / "prelude_out"), str(score)])
        midi = tmp_path / "prelude_out.midi"
        # Delegate MusicXML write to inline music21 convert (same as sandbox).
        from music21 import converter, stream, clef, meter, key, tempo, metadata
        from music21.musicxml.m21ToXml import GeneralObjectExporter

        s = converter.parse(str(midi))
        out = stream.Score()
        part = stream.Part()
        part.insert(0, clef.BassClef())
        part.insert(0, key.KeySignature(1))
        part.insert(0, meter.TimeSignature("4/4"))
        part.insert(0, tempo.MetronomeMark(number=80))
        md = metadata.Metadata()
        md.title = "Bach Cello Suite No. 1, BWV 1007 — Prélude"
        md.composer = "Johann Sebastian Bach"
        out.insert(0, md)
        midi_part = s.parts[0] if s.parts else s
        for el in midi_part.flatten().notesAndRests:
            part.insert(el.offset, el)
        out.append(part)
        xml = GeneralObjectExporter(out).parse()
        if isinstance(xml, bytes):
            xml = xml.decode("utf-8")
        xml = xml.replace(
            "<identification>",
            "<identification>\n    <rights>Public Domain — Mutopia-2018/01/19-517 after "
            "Schirmer, 1916. Derived from LilyPond-compiled Mutopia MIDI "
            "(https://www.mutopiaproject.org/cgibin/piece-info.cgi?id=517).</rights>",
            1,
        )
        _OUT.write_text(xml, encoding="utf-8")
        print(f"Wrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
