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


def _iter_pitched_notes(root: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    """Return ``(measure, note)`` for monophonic fingerprint notes (no rests/chords)."""
    out: list[tuple[ET.Element, ET.Element]] = []
    for measure in root.iter():
        if _local(measure.tag) != "measure":
            continue
        for note in measure:
            if _local(note.tag) != "note":
                continue
            is_rest = any(_local(c.tag) == "rest" for c in note)
            is_chord = any(_local(c.tag) == "chord" for c in note)
            if is_rest or is_chord:
                continue
            if any(_local(c.tag) == "pitch" for c in note):
                out.append((measure, note))
    return out


def _measure_number(measure: ET.Element) -> int | None:
    raw = measure.get("number")
    if raw is None:
        return None
    try:
        return int(str(raw).split(".")[0])
    except ValueError:
        return None


def resolve_clip_note_span(
    *,
    ref_notes: int,
    hyp_notes: int,
    window_offset: int,
    window_mode: str,
    audio_duration_sec: float | None = None,
    typical_duration_sec: float | None = None,
    coverage_keep_full: float = 0.85,
) -> tuple[int, int] | None:
    """Decide which edition note indices to keep for a clip-length MATCH export.

    Returns ``(start, end)`` half-open indices into the edition fingerprint, or
    ``None`` when the whole movement should be kept (clip ≈ full piece).

    Never expands to a multi-work collection — callers must already point at a
    single-movement catalog entry (prefer Caprice_No._2.pdf over Complete).
    """
    if ref_notes <= 0 or hyp_notes <= 0:
        return None

    # Clip covers most of this movement → keep the whole Caprice / movement.
    if hyp_notes >= coverage_keep_full * ref_notes:
        return None
    if (
        audio_duration_sec is not None
        and typical_duration_sec is not None
        and typical_duration_sec > 0
        and audio_duration_sec >= coverage_keep_full * typical_duration_sec
    ):
        return None

    if window_mode == "hyp_in_ref":
        start = max(0, int(window_offset))
        end = min(ref_notes, start + int(hyp_notes))
        if end - start < 8:
            return None
        # Near-full span after clipping noise → keep whole movement.
        if (end - start) >= coverage_keep_full * ref_notes:
            return None
        return start, end

    # ref_in_hyp or unknown: take a duration-proportional prefix when audio
    # is clearly shorter than the catalog movement.
    if (
        audio_duration_sec is not None
        and typical_duration_sec is not None
        and typical_duration_sec > 0
        and audio_duration_sec < coverage_keep_full * typical_duration_sec
    ):
        frac = max(0.05, min(1.0, audio_duration_sec / typical_duration_sec))
        end = max(8, min(ref_notes, int(round(ref_notes * frac))))
        if end >= coverage_keep_full * ref_notes:
            return None
        return 0, end

    return None


def excerpt_musicxml_by_notes(
    edition_path: Path,
    dest: Path,
    *,
    start_note: int,
    end_note: int,
) -> Path:
    """Write a MusicXML that keeps only measures spanning ``[start_note, end_note)``.

    Pitch content outside the clip span is removed at measure granularity so
    MATCH exports the **played** Caprice / phrase, not an entire collection.
    Attributes (clef/key/time) from the first kept measure are retained.
    """
    edition_path = Path(edition_path)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    xml_path = resolve_edition_path(edition_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pitched = _iter_pitched_notes(root)
    if not pitched:
        shutil.copy2(xml_path, dest)
        return dest

    start_note = max(0, int(start_note))
    end_note = min(len(pitched), int(end_note))
    if start_note >= end_note:
        shutil.copy2(xml_path, dest)
        return dest

    first_meas = pitched[start_note][0]
    last_meas = pitched[end_note - 1][0]
    first_num = _measure_number(first_meas)
    last_num = _measure_number(last_meas)

    # Collect measure elements to keep (by identity first, then by number range).
    keep_ids = {id(pitched[i][0]) for i in range(start_note, end_note)}
    if first_num is not None and last_num is not None:
        for measure in root.iter():
            if _local(measure.tag) != "measure":
                continue
            n = _measure_number(measure)
            if n is not None and first_num <= n <= last_num:
                keep_ids.add(id(measure))

    # Drop measures outside the span from each <part>.
    for part in root.iter():
        if _local(part.tag) != "part":
            continue
        for child in list(part):
            if _local(child.tag) == "measure" and id(child) not in keep_ids:
                part.remove(child)

    # Renumber remaining measures from 1 for MuseScore friendliness.
    for part in root.iter():
        if _local(part.tag) != "part":
            continue
        num = 1
        for child in part:
            if _local(child.tag) != "measure":
                continue
            child.set("number", str(num))
            num += 1

    # Annotate identification so MuseScore / diffs show this is a clip excerpt.
    ident = None
    for el in root:
        if _local(el.tag) == "identification":
            ident = el
            break
    if ident is None:
        ident = ET.Element("identification")
        # Insert after movement-title / work when present.
        root.insert(0, ident)
    misc = ET.SubElement(ident, "miscellaneous")
    field = ET.SubElement(misc, "miscellaneous-field", name="cnd_clip_excerpt")
    field.text = f"notes[{start_note}:{end_note}] of {len(pitched)}"

    # ElementTree does not preserve DOCTYPE; MuseScore still opens UTF-8 MusicXML.
    tree.write(dest, encoding="utf-8", xml_declaration=True)
    return dest


def write_match_musicxml(
    edition_path: Path,
    dest: Path,
    *,
    start_note: int | None = None,
    end_note: int | None = None,
) -> tuple[Path, str]:
    """Copy or excerpt an edition for MATCH attach.

    Returns ``(path, kind)`` where kind is ``verbatim_pd_edition`` or
    ``clip_excerpt_pd_edition``.
    """
    if start_note is None or end_note is None:
        return copy_edition_musicxml(edition_path, dest), "verbatim_pd_edition"
    written = excerpt_musicxml_by_notes(
        edition_path, dest, start_note=start_note, end_note=end_note
    )
    return written, "clip_excerpt_pd_edition"
