"""Export fused AV notes to bass-clef MusicXML with fingerings + positions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cello_note_detector.audio.load import load_mono
from cello_note_detector.decode.events import NoteEvent
from cello_note_detector.decode.key import infer_key
from cello_note_detector.decode.rhythm import analyze_rhythm, resolve_quantize_preset
from cello_note_detector.export.musicxml import write_musicxml
from cello_note_detector.video.position_map import (
    POSITION_MAPPING_DOC,
    annotate_fused_note,
)


def _fused_to_note_events(fused_notes: list[dict[str, Any]]) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    for n in fused_notes:
        midi = int(n["midi"])
        onset = float(n["onset"])
        offset = float(n.get("offset", onset + float(n.get("duration", 0.1))))
        duration = max(offset - onset, 1e-3)
        events.append(
            NoteEvent(
                pitch_hz=float(n.get("pitch_hz") or 0.0),
                midi=midi,
                note=str(n.get("note") or f"MIDI{midi}"),
                onset=onset,
                offset=offset,
                duration=duration,
                confidence=float(n.get("confidence") or 0.5),
            )
        )
    return events


def _mark_position_changes(annotated: list[dict[str, Any]]) -> None:
    """Set ``score_position_mark`` on first note and whenever position changes."""
    prev: str | None = None
    for note in annotated:
        label = note.get("score_position")
        if not label:
            note["score_position_mark"] = None
            continue
        if prev is None or label != prev:
            note["score_position_mark"] = label
            prev = label
        else:
            note["score_position_mark"] = None


def build_fused_score_result(
    fused: dict[str, Any],
    *,
    audio_path: Path | None = None,
    quantize: str = "coarse",
    title: str | None = None,
) -> dict[str, Any]:
    """Rhythm-quantize fused notes and attach fingering / position for MusicXML.

    General for any monophonic cello fusion timeline (not clip-specific).
    """
    fused_notes = list(fused.get("fused_notes") or [])
    if not fused_notes:
        raise ValueError("No fused_notes to export — run --fuse first.")

    annotated = [annotate_fused_note(n) for n in fused_notes]
    _mark_position_changes(annotated)

    wav = Path(audio_path or fused.get("audio_path") or "")
    if not wav.is_file():
        # Fall back to silent placeholder audio sized to last offset so tempo
        # estimation can still use note onsets.
        last = max(float(n.get("offset") or 0.0) for n in annotated)
        sr = 16000
        y = np.zeros(max(int(last * sr) + sr, sr), dtype=np.float32)
    else:
        y, sr = load_mono(wav)

    note_events = _fused_to_note_events(annotated)
    qcfg = resolve_quantize_preset(quantize)
    rhythm = analyze_rhythm(
        y,
        note_events,
        sr=sr,
        grid_beats=float(qcfg["grid_beats"]),
        min_rest_sec=float(qcfg["min_rest_sec"]),
        min_rest_beats=float(qcfg["min_rest_beats"]),
        min_note_beats=float(qcfg["min_note_beats"]),
        catalog=qcfg["catalog"],
        score_intent=bool(qcfg.get("score_intent", False)),
    )

    # Merge rhythm fields onto annotated notes (parallel order).
    notes_out: list[dict[str, Any]] = []
    for note, nr in zip(annotated, rhythm["note_rhythm"]):
        item = {
            "type": "note",
            "midi": note["midi"],
            "note": note.get("note"),
            "onset": note["onset"],
            "offset": note["offset"],
            "duration": note.get("duration"),
            "confidence": note.get("confidence"),
            "string": note.get("score_string"),
            "fingering": note.get("fingering"),
            "position_band": note.get("position_band"),
            "semitones_above_open": note.get("semitones_above_open"),
            "score_finger": note.get("score_finger"),
            "score_position": note.get("score_position"),
            "score_position_mark": note.get("score_position_mark"),
            "score_string_number": note.get("score_string_number"),
        }
        item.update(nr)
        notes_out.append(item)

    # Chronological events: notes get score_* fields; rests do not.
    note_by_onset = {(float(n["onset"]), int(n["midi"])): n for n in notes_out}
    timeline: list[dict[str, Any]] = []
    for ev in rhythm["events"]:
        if ev.get("type") == "note" or ("midi" in ev and ev.get("type") != "rest"):
            key = (float(ev["onset"]), int(ev["midi"]))
            rich = note_by_onset.get(key)
            if rich is not None:
                merged = dict(ev)
                merged.update(
                    {
                        "score_finger": rich.get("score_finger"),
                        "score_position": rich.get("score_position"),
                        "score_position_mark": rich.get("score_position_mark"),
                        "score_string_number": rich.get("score_string_number"),
                        "string": rich.get("string"),
                    }
                )
                timeline.append(merged)
            else:
                timeline.append(dict(ev))
        else:
            timeline.append(dict(ev))

    key_info = infer_key(note_events)
    result: dict[str, Any] = {
        "schema": "cello_fused_score_v0",
        "source": fused.get("source"),
        "audio_path": str(wav) if wav else fused.get("audio_path"),
        "video_path": fused.get("video_path"),
        "title": title,
        "tempo": rhythm["tempo"],
        "key": key_info,
        "notes": notes_out,
        "rests": rhythm["rests"],
        "events": timeline,
        "num_notes": len(notes_out),
        "num_rests": rhythm["num_rests"],
        "rhythm": {
            "quantize": qcfg["name"],
            "grid_beats": rhythm["grid_beats"],
        },
        "position_mapping": POSITION_MAPPING_DOC,
        "limits": [
            "Fingerings and positions inferred from AV fusion — not edition fingerings.",
            "Bow up/down marks and shift lines not yet engraved.",
            "Low-confidence fingerings are still written (no confidence filter yet).",
        ],
    }
    return result


def write_fused_musicxml(
    fused: dict[str, Any],
    path: Path,
    *,
    audio_path: Path | None = None,
    quantize: str = "coarse",
    title: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write ``*_fused.musicxml`` and return ``(path, score_result)``."""
    score = build_fused_score_result(
        fused,
        audio_path=audio_path,
        quantize=quantize,
        title=title,
    )
    out = Path(path)
    write_musicxml(
        score,
        out,
        title=title or score.get("title") or Path(out).stem,
        part_name="Cello",
    )
    return out, score
