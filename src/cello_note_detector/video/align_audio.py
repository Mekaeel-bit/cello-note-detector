"""Optional rough audio alignment: extract WAV and attach pyin note onsets."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any


def extract_wav(video_path: Path, wav_path: Path, *, sr: int = 16000) -> Path:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return wav_path


def attach_audio_notes(timeline: dict[str, Any], video_path: Path) -> dict[str, Any]:
    """Best-effort: add `audio_notes` from the existing pyin pipeline."""
    try:
        from cello_note_detector.audio.load import load_mono
        from cello_note_detector.audio.pitch_baseline import transcribe_notes
        from cello_note_detector.decode.events import note_events_to_dicts
    except Exception as exc:  # pragma: no cover
        timeline["audio_alignment"] = {
            "ok": False,
            "error": f"audio deps unavailable: {exc}",
        }
        return timeline

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        try:
            extract_wav(video_path, wav)
        except subprocess.CalledProcessError as exc:
            timeline["audio_alignment"] = {
                "ok": False,
                "error": f"ffmpeg extract failed: {exc}",
            }
            return timeline

        y, sr = load_mono(wav)
        notes, _meta = transcribe_notes(y, sr=sr)
        note_dicts = note_events_to_dicts(notes)

    timeline["audio_alignment"] = {
        "ok": True,
        "method": "ffmpeg-extract + pyin",
        "num_notes": len(note_dicts),
        "note": "Rough sync by shared clock; no AV drift correction.",
    }
    timeline["audio_notes"] = note_dicts

    # Cross-link nearest bow/finger events to each note onset (±0.35s)
    events = timeline.get("events") or []
    for n in note_dicts:
        onset = float(n.get("onset", 0.0))
        nearby = [e for e in events if abs(float(e["t"]) - onset) <= 0.35]
        n["nearby_video_events"] = nearby
    return timeline
