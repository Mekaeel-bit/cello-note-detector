"""Read PD MusicXML/MXL editions: fingerprints, ornaments, verbatim copy."""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _step_alter_to_pc(step: str, alter: int) -> int:
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step.upper()]
    return (base + int(alter)) % 12


def resolve_edition_path(path: Path) -> Path:
    """Return a usable MusicXML path (extracts ``.mxl`` to a sibling folder)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".mxl":
        return path
    extract_dir = path.with_suffix(".mxl_extracted")
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
    # Prefer container score XML.
    candidates = list(extract_dir.rglob("*.xml")) + list(
        extract_dir.rglob("*.musicxml")
    )
    if not candidates:
        raise FileNotFoundError(f"No MusicXML inside MXL: {path}")
    # Skip META-INF container if possible.
    for c in candidates:
        if "META-INF" not in c.parts and c.name.lower() != "container.xml":
            return c
    return candidates[0]


def parse_musicxml_fingerprint(path: Path) -> dict[str, Any]:
    """Extract match fingerprint + ornament stats from an edition MusicXML/MXL."""
    xml_path = resolve_edition_path(path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    midi_sequence: list[int] = []
    slur_count = 0
    trill_count = 0
    tie_count = 0
    measure_count = 0
    title = None
    for el in root.iter():
        name = _local(el.tag)
        if name == "work-title" and el.text and title is None:
            title = el.text.strip()
        if name == "movement-title" and el.text and title is None:
            title = el.text.strip()
        if name == "measure":
            measure_count += 1
        if name == "slur":
            slur_count += 1
        if name in {"trill-mark", "wavy-line", "inverted-mordent", "mordent", "turn"}:
            trill_count += 1
        if name == "tie":
            tie_count += 1
        if name != "note":
            continue
        is_rest = any(_local(c.tag) == "rest" for c in el)
        is_chord = any(_local(c.tag) == "chord" for c in el)
        if is_rest or is_chord:
            # Skip chord tones for monophonic fingerprint (keep first voice note).
            continue
        step = alter = octave = None
        for c in el:
            if _local(c.tag) != "pitch":
                continue
            for p in c:
                pn = _local(p.tag)
                if pn == "step" and p.text:
                    step = p.text.strip()
                elif pn == "alter" and p.text:
                    alter = int(float(p.text))
                elif pn == "octave" and p.text:
                    octave = int(p.text)
        if step is None or octave is None:
            continue
        alter = alter or 0
        pc = _step_alter_to_pc(step, alter)
        midi = (octave + 1) * 12 + pc
        midi_sequence.append(midi)

    return {
        "title": title,
        "source_musicxml": str(path),
        "resolved_musicxml": str(xml_path),
        "num_notes": len(midi_sequence),
        "num_measures": measure_count,
        "midi_sequence": midi_sequence,
        "ornaments": {
            "slurs": slur_count,
            "trills_or_waves": trill_count,
            "ties": tie_count,
            "has_slurs": slur_count > 0,
            "has_trills": trill_count > 0,
        },
    }


def copy_edition_musicxml(
    edition_path: Path,
    dest: Path,
) -> Path:
    """Copy a PD edition MusicXML/MXL to ``dest`` without rewriting pitches.

    ``.mxl`` is extracted to a ``.musicxml`` beside ``dest`` when needed so
    MuseScore opens a plain MusicXML path. Markings (slurs/trills/etc.) are
    preserved byte-for-byte from the edition file.
    """
    edition_path = Path(edition_path)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if edition_path.suffix.lower() == ".mxl":
        # Extract primary score XML and write as .musicxml for MuseScore.
        xml_path = resolve_edition_path(edition_path)
        if dest.suffix.lower() == ".mxl":
            shutil.copy2(edition_path, dest)
            # Also write an unzipped musicxml next to it for convenience.
            side = dest.with_suffix(".musicxml")
            shutil.copy2(xml_path, side)
            return side
        shutil.copy2(xml_path, dest)
        return dest
    shutil.copy2(edition_path, dest)
    return dest


def estimate_movements_by_duration(
    movements: list[dict[str, Any]],
    audio_duration_sec: float,
) -> dict[str, Any] | None:
    """Pick the catalog movement whose typical duration best matches the audio.

    Each movement dict may include ``typical_duration_sec``. Returns the best
    movement or ``None`` if none declare durations.
    """
    scored = []
    for mov in movements:
        typ = mov.get("typical_duration_sec")
        if typ is None:
            continue
        scored.append((abs(float(typ) - float(audio_duration_sec)), mov))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1]
