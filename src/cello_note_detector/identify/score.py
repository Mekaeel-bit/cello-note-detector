"""Attach verbatim PD MusicXML editions on catalog match."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .catalog import CatalogEntry
from .match import IdentificationResult
from .musicxml_edition import copy_edition_musicxml, estimate_movements_by_duration


def _likely_from_duration_key(
    candidates: list,
    catalog_entries: list[CatalogEntry],
    *,
    duration_sec: float,
    key_name: str,
) -> dict[str, Any] | None:
    """Soft hint when pitch MATCH fails but duration/key align.

    Never sets matched=true — UX / debugging only.
    """
    if not candidates or not catalog_entries:
        return None
    entry_map = {e.id: e for e in catalog_entries}
    key_l = key_name.lower().replace(" ", "")
    scored: list[tuple[float, Any, CatalogEntry]] = []
    for c in candidates:
        entry = entry_map.get(c.entry_id)
        if entry is None:
            continue
        score = float(c.confidence)
        if (
            duration_sec > 0
            and entry.typical_duration_sec is not None
            and entry.typical_duration_sec > 0
        ):
            rel = abs(duration_sec - entry.typical_duration_sec) / entry.typical_duration_sec
            if rel <= 0.12:
                score += 0.25
            elif rel <= 0.25:
                score += 0.1
        if key_l and entry.key:
            ek = entry.key.lower().replace(" ", "")
            if ek in key_l or key_l in ek:
                score += 0.2
        scored.append((score, c, entry))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_c, best_e = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 0.4 or (best_score - second) < 0.08:
        return None
    return {
        "piece_id": best_e.id,
        "title": best_e.title,
        "composer": best_e.composer,
        "pitch_confidence": round(float(best_c.confidence), 4),
        "soft_score": round(best_score, 4),
        "reason": "duration_key_align_not_pitch_match",
        "note": (
            "Soft hint only — not a MATCH. Pitch gates failed; open "
            "transcription MusicXML, or re-record / confirm Caprice number."
        ),
    }


def attach_catalog_score(
    result: dict[str, Any],
    identification: IdentificationResult,
    *,
    output_dir: Path,
    stem: str,
    export_musicxml: bool,
    export_midi: bool,
    bpm: float | None = None,
    catalog_entries: list[CatalogEntry] | None = None,
) -> dict[str, Any]:
    """Copy the published MusicXML edition into ``output_dir`` (verbatim).

    Never regenerates pitches from JSON/MIDI for the primary MuseScore file.
    On no match, records fallback metadata and leaves transcription exports alone.
    """
    result["identification"] = identification.to_metadata()
    result["score_source"] = identification.score_source

    if not identification.matched or identification.entry is None:
        result["identification"]["fallback"] = "audio_transcription"
        duration = float(result.get("duration_sec") or 0.0)
        key_name = ""
        key_info = result.get("key")
        if isinstance(key_info, dict):
            key_name = str(key_info.get("name") or "")
        likely = _likely_from_duration_key(
            identification.candidates,
            catalog_entries or [],
            duration_sec=duration,
            key_name=key_name,
        )
        if likely:
            result["identification"]["likely"] = likely
        return result

    entry = identification.entry

    # Optional: if this id is a multi-movement collection, prefer the movement
    # whose typical duration is closest to the audio (pitch still from edition).
    if catalog_entries and entry.typical_duration_sec is not None:
        duration = float(result.get("duration_sec") or 0.0)
        movs = [
            {
                "entry": e,
                "typical_duration_sec": e.typical_duration_sec,
            }
            for e in catalog_entries
            if e.typical_duration_sec is not None
            and e.id.startswith(entry.id.split("_")[0])
        ]
        # Only re-pick among siblings when several duration-tagged entries exist.
        if len(movs) > 1 and duration > 0:
            pick = estimate_movements_by_duration(movs, duration)
            if pick is not None and pick["entry"].id != entry.id:
                entry = pick["entry"]
                identification.entry = entry
                result["identification"] = identification.to_metadata()
                result["identification"]["duration_selected"] = True

    result["identification"]["edition_musicxml_source"] = str(entry.edition_musicxml)
    if entry.ornaments:
        result["identification"]["edition_ornaments"] = entry.ornaments

    # Sidecar copies for debugging / provenance.
    if entry.fingerprint_json and entry.fingerprint_json.is_file():
        fp_copy = output_dir / f"{stem}_catalog_fingerprint.json"
        shutil.copy2(entry.fingerprint_json, fp_copy)
        result["identification"]["catalog_fingerprint_copy"] = str(fp_copy)
    if entry.midi_path and entry.midi_path.is_file():
        mid_copy = output_dir / f"{stem}_catalog_source.mid"
        shutil.copy2(entry.midi_path, mid_copy)
        result["identification"]["catalog_midi_copy"] = str(mid_copy)

    if export_musicxml:
        xml_path = output_dir / f"{stem}.musicxml"
        written = copy_edition_musicxml(entry.edition_musicxml, xml_path)
        result["_musicxml_path"] = str(written)
        result["identification"]["score_musicxml"] = str(written)
        result["identification"]["score_kind"] = "verbatim_pd_edition"
        result["identification"]["score_note_count"] = entry.num_notes

    if export_midi and entry.midi_path and entry.midi_path.is_file():
        # Prefer publisher MIDI when present; otherwise skip (don't invent MIDI).
        mid_path = output_dir / f"{stem}.mid"
        shutil.copy2(entry.midi_path, mid_path)
        result["_midi_path"] = str(mid_path)
        result["identification"]["score_midi"] = str(mid_path)

    result["catalog_score"] = {
        "piece_id": entry.id,
        "title": entry.title,
        "edition": entry.edition,
        "license": entry.license,
        "num_notes": entry.num_notes,
        "ornaments": entry.ornaments,
        "edition_musicxml": str(entry.edition_musicxml),
    }

    meta_path = output_dir / f"{stem}_identification.json"
    result["identification"]["metadata_path"] = str(meta_path)
    meta_path.write_text(
        json.dumps(result["identification"], indent=2),
        encoding="utf-8",
    )
    return result
