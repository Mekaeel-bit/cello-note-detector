"""Attach / generate published scores from a matched catalog entry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from cello_note_detector.decode.key import parse_key
from cello_note_detector.export import write_midi, write_musicxml

from .catalog import CatalogEntry
from .match import IdentificationResult

# Maps duration_value → beats (mirrors export/musicxml defaults).
_DURATION_BEATS: dict[str, float] = {
    "whole": 4.0,
    "dotted_half": 3.0,
    "half": 2.0,
    "dotted_quarter": 1.5,
    "quarter": 1.0,
    "dotted_eighth": 0.75,
    "eighth": 0.5,
    "dotted_sixteenth": 0.375,
    "sixteenth": 0.25,
    "thirty_second": 0.125,
}


def _midi_to_note_name(midi: int) -> str:
    names = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
    return f"{names[midi % 12]}{midi // 12 - 1}"


def _key_info(entry: CatalogEntry) -> dict[str, Any]:
    raw = entry.key or (entry.reference.get("key") if entry.reference else None)
    if raw:
        try:
            return parse_key(str(raw))
        except ValueError:
            pass
    return {
        "name": str(raw or "C major"),
        "tonic": "C",
        "mode": "major",
        "fifths": 0,
        "method": "catalog",
        "confidence": 1.0,
    }


def reference_to_result(
    entry: CatalogEntry,
    *,
    audio_path: str | None = None,
    bpm: float = 60.0,
) -> dict[str, Any]:
    """Build a CLI-shaped result dict from a catalog reference for export.

    Uses written pitch/rhythm from the Mutopia/Schirmer (or other PD) reference
    rather than the performance transcription.
    """
    ref = entry.reference
    events_in = ref.get("events") or []
    notes: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    beat_sec = 60.0 / max(float(bpm), 1e-6)

    for ev in events_in:
        if "midi" not in ev:
            continue
        midi = int(ev["midi"])
        dur_beats = float(
            ev.get("duration_beats")
            or _DURATION_BEATS.get(str(ev.get("duration_value") or ""), 0.25)
        )
        onset_beats = float(ev.get("onset_beats", 0.0))
        dur_name = ev.get("duration_value")
        if not dur_name:
            # Nearest catalog name.
            dur_name = min(
                _DURATION_BEATS.items(),
                key=lambda kv: abs(kv[1] - dur_beats),
            )[0]
        onset_sec = onset_beats * beat_sec
        dur_sec = dur_beats * beat_sec
        note = {
            "midi": midi,
            "note": _midi_to_note_name(midi),
            "onset": round(onset_sec, 4),
            "offset": round(onset_sec + dur_sec, 4),
            "duration": round(dur_sec, 4),
            "confidence": 1.0,
            "onset_beats": onset_beats,
            "duration_beats": dur_beats,
            "quantized_onset_beats": onset_beats,
            "quantized_duration_beats": dur_beats,
            "duration_value": dur_name,
            "duration_value_beats": dur_beats,
            "source": "catalog_reference",
        }
        notes.append(note)
        timeline.append({**note, "type": "note"})

    key_info = _key_info(entry)
    key_info = dict(key_info)
    key_info["method"] = "catalog"
    key_info["confidence"] = 1.0

    total_beats = float(ref.get("total_beats") or (notes[-1]["onset_beats"] + notes[-1]["duration_beats"] if notes else 0.0))
    return {
        "audio_path": audio_path,
        "sample_rate": None,
        "duration_sec": round(total_beats * beat_sec, 4),
        "pitch_method": "catalog_reference",
        "score_source": "catalog",
        "catalog_piece_id": entry.id,
        "key": key_info,
        "tempo": {
            "bpm": float(bpm),
            "method": "catalog_default",
            "note": "Catalog score uses a nominal BPM for MusicXML/MIDI playback; written durations come from the PD reference.",
        },
        "num_notes": len(notes),
        "notes": notes,
        "num_rests": 0,
        "rests": [],
        "events": timeline,
        "rhythm": {
            "quantize": "catalog",
            "score_intent": True,
            "method": "published public-domain reference (local catalog)",
        },
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
) -> dict[str, Any]:
    """Copy/generate the published score into ``output_dir`` and annotate JSON.

    Always records identification metadata. When matched, writes catalog-based
    MusicXML/MIDI (and a sidecar reference copy). When not matched, leaves the
    existing transcription exports untouched.
    """
    result["identification"] = identification.to_metadata()
    result["score_source"] = identification.score_source

    if not identification.matched or identification.entry is None:
        result["identification"]["fallback"] = "audio_transcription"
        return result

    entry = identification.entry
    # Prefer performance-estimated tempo for playback if available.
    tempo_bpm = bpm
    if tempo_bpm is None:
        tempo = result.get("tempo")
        if isinstance(tempo, dict) and tempo.get("bpm"):
            tempo_bpm = float(tempo["bpm"])
        else:
            tempo_bpm = 60.0

    catalog_result = reference_to_result(
        entry,
        audio_path=result.get("audio_path"),
        bpm=float(tempo_bpm),
    )

    # Sidecar: copy the monophonic reference JSON next to outputs.
    ref_copy = output_dir / f"{stem}_catalog_reference.json"
    shutil.copy2(entry.reference_json, ref_copy)
    result["identification"]["catalog_reference_copy"] = str(ref_copy)

    # Optional: copy Mutopia MIDI if present.
    if entry.midi_path and entry.midi_path.is_file():
        mid_copy = output_dir / f"{stem}_catalog_source.mid"
        shutil.copy2(entry.midi_path, mid_copy)
        result["identification"]["catalog_midi_copy"] = str(mid_copy)

    if export_musicxml:
        xml_path = output_dir / f"{stem}.musicxml"
        write_musicxml(catalog_result, xml_path, title=entry.title)
        result["_musicxml_path"] = str(xml_path)
        result["identification"]["score_musicxml"] = str(xml_path)
        result["identification"]["score_kind"] = "catalog_reference"

    if export_midi:
        mid_path = output_dir / f"{stem}.mid"
        write_midi(catalog_result, mid_path)
        result["_midi_path"] = str(mid_path)
        result["identification"]["score_midi"] = str(mid_path)
        result["identification"]["score_kind"] = "catalog_reference"

    # Keep the performance transcription notes in the JSON, but mark that the
    # sheet artifacts came from the catalog when exported.
    result["catalog_score"] = {
        "piece_id": entry.id,
        "title": entry.title,
        "edition": entry.edition,
        "num_notes": catalog_result["num_notes"],
        "key": catalog_result["key"],
        "tempo_bpm": catalog_result["tempo"]["bpm"],
    }

    # Write match metadata last so score paths / score_kind are included.
    meta_path = output_dir / f"{stem}_identification.json"
    result["identification"]["metadata_path"] = str(meta_path)
    meta_path.write_text(
        json.dumps(result["identification"], indent=2),
        encoding="utf-8",
    )
    return result
