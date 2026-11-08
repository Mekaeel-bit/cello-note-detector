"""Attach verbatim / clip-excerpt PD MusicXML editions on catalog match."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .catalog import CatalogEntry
from .match import IdentificationResult, MatchScore
from .musicxml_edition import (
    estimate_movements_by_duration,
    resolve_clip_note_span,
    write_match_musicxml,
)

# Soft duration/key ranking → hard MATCH when pitch gates fail but evidence
# is strong (e.g. Caprices clip: G minor + ~176s → Caprice 2 OMR edition).
_SOFT_HINT_MIN = 0.4
_SOFT_HINT_MARGIN = 0.08
# Caprices clip lands ~0.72 soft; require clear duration+key + margin so we
# do not attach Caprice 2 to arbitrary G-minor takes.
_HARD_PROMOTE_MIN = 0.60
_HARD_PROMOTE_MARGIN = 0.10
_HARD_PROMOTE_DURATION_REL = 0.12


def _duration_key_scores(
    candidates: list,
    catalog_entries: list[CatalogEntry],
    *,
    duration_sec: float,
    key_name: str,
) -> list[tuple[float, Any, CatalogEntry, dict[str, Any]]]:
    """Rank catalog candidates by pitch conf + duration/key bonuses."""
    if not candidates or not catalog_entries:
        return []
    entry_map = {e.id: e for e in catalog_entries}
    key_l = (key_name or "").lower().replace(" ", "")
    scored: list[tuple[float, Any, CatalogEntry, dict[str, Any]]] = []
    for c in candidates:
        entry = entry_map.get(c.entry_id)
        if entry is None:
            continue
        score = float(c.confidence)
        dur_rel: float | None = None
        dur_bonus = 0.0
        key_hit = False
        if (
            duration_sec > 0
            and entry.typical_duration_sec is not None
            and entry.typical_duration_sec > 0
        ):
            dur_rel = abs(duration_sec - entry.typical_duration_sec) / entry.typical_duration_sec
            if dur_rel <= 0.12:
                dur_bonus = 0.25
            elif dur_rel <= 0.25:
                dur_bonus = 0.1
            score += dur_bonus
        if key_l and entry.key:
            ek = entry.key.lower().replace(" ", "")
            if ek in key_l or key_l in ek:
                key_hit = True
                score += 0.2
        scored.append(
            (
                score,
                c,
                entry,
                {
                    "duration_rel": dur_rel,
                    "duration_bonus": dur_bonus,
                    "key_match": key_hit,
                },
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _likely_from_duration_key(
    candidates: list,
    catalog_entries: list[CatalogEntry],
    *,
    duration_sec: float,
    key_name: str,
) -> dict[str, Any] | None:
    """Soft hint when pitch MATCH fails but duration/key align (debug UX)."""
    scored = _duration_key_scores(
        candidates,
        catalog_entries,
        duration_sec=duration_sec,
        key_name=key_name,
    )
    if not scored:
        return None
    best_score, best_c, best_e, _meta = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < _SOFT_HINT_MIN or (best_score - second) < _SOFT_HINT_MARGIN:
        return None
    return {
        "piece_id": best_e.id,
        "title": best_e.title,
        "composer": best_e.composer,
        "pitch_confidence": round(float(best_c.confidence), 4),
        "soft_score": round(best_score, 4),
        "reason": "duration_key_align_not_pitch_match",
        "note": (
            "Soft hint only — not a MATCH. Pitch gates failed and evidence "
            "was below the hard-promote threshold."
        ),
    }


def promote_soft_catalog_match(
    identification: IdentificationResult,
    catalog_entries: list[CatalogEntry] | None,
    *,
    duration_sec: float,
    key_name: str,
) -> IdentificationResult:
    """Upgrade a strong duration/key soft hit into a hard catalog MATCH.

    Pitch-contiguous gates often fail on virtuosic Caprices (pyin / double
    stops), while key + typical duration already uniquely pick Caprice 2 and
    we have an OMR/IMSLP MusicXML edition. Prefer that edition over a garbage
    transcription MuseScore dump.

    Requires: soft_score ≥ 0.65, margin ≥ 0.12, key match, tight duration
    (≤12% rel), and a readable ``edition_musicxml`` on disk.
    """
    if identification.matched and identification.entry is not None:
        return identification
    if not catalog_entries:
        return identification

    scored = _duration_key_scores(
        identification.candidates,
        catalog_entries,
        duration_sec=duration_sec,
        key_name=key_name,
    )
    if not scored:
        return identification

    best_score, best_c, best_e, meta = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    dur_rel = meta.get("duration_rel")
    if not meta.get("key_match"):
        return identification
    if dur_rel is None or float(dur_rel) > _HARD_PROMOTE_DURATION_REL:
        return identification
    if best_score < _HARD_PROMOTE_MIN or (best_score - second) < _HARD_PROMOTE_MARGIN:
        return identification
    if not best_e.edition_musicxml.is_file():
        return identification

    best_ms: MatchScore | None = best_c if isinstance(best_c, MatchScore) else None
    return IdentificationResult(
        matched=True,
        confidence=float(best_score),
        reason="duration_key_catalog_match",
        entry=best_e,
        best=best_ms,
        candidates=identification.candidates,
        score_source="catalog",
    )


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
    if identification.reason == "duration_key_catalog_match":
        result["identification"]["match_mode"] = "duration_key_soft_promote"
        result["identification"]["match_note"] = (
            "Pitch-contiguous gates failed; hard MATCH from duration+key "
            "alignment against a catalog edition (OMR/IMSLP MusicXML). "
            "Primary MuseScore file is the PD edition, not transcription."
        )
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
        # MATCH must ship only the played clip span (e.g. Capriccio No.2),
        # never an entire Caprices collection. Prefer per-movement editions;
        # when the hyp is a mid-piece excerpt, trim measures to that span.
        span = None
        if identification.best is not None:
            span = resolve_clip_note_span(
                ref_notes=int(identification.best.ref_notes or entry.num_notes),
                hyp_notes=int(identification.best.hyp_notes or 0),
                window_offset=int(identification.best.window_offset or 0),
                window_mode=str(identification.best.window_mode or ""),
                audio_duration_sec=float(result.get("duration_sec") or 0.0) or None,
                typical_duration_sec=entry.typical_duration_sec,
            )
        start_n = span[0] if span else None
        end_n = span[1] if span else None
        written, score_kind = write_match_musicxml(
            entry.edition_musicxml,
            xml_path,
            start_note=start_n,
            end_note=end_n,
        )
        result["_musicxml_path"] = str(written)
        result["identification"]["score_musicxml"] = str(written)
        result["identification"]["score_kind"] = score_kind
        if span is not None:
            result["identification"]["clip_excerpt"] = {
                "start_note": start_n,
                "end_note": end_n,
                "ref_notes": entry.num_notes,
                "note": (
                    "Exported only the clip-aligned measures from the PD "
                    "edition (not a multi-work collection)."
                ),
            }
            result["identification"]["score_note_count"] = max(0, end_n - start_n)
        else:
            result["identification"]["score_note_count"] = entry.num_notes
            result["identification"]["clip_excerpt"] = {
                "full_movement": True,
                "note": (
                    "Clip covers this single movement; exported the whole "
                    "movement MusicXML (still not a Caprices collection)."
                ),
            }

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
