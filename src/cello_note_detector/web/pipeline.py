"""Run the AV-fuse → MusicXML pipeline for a single uploaded video."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from cello_note_detector.video.align_audio import SilentAudioError, attach_audio_notes
from cello_note_detector.video.detect import analyze_video, write_timeline
from cello_note_detector.video.fusion import fuse_timeline
from cello_note_detector.video.models import DEFAULT_MODEL_DIR
from cello_note_detector.video.overlay_fuse import write_fused_overlays
from cello_note_detector.video.sheet import write_fused_musicxml

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


def process_video_to_score(
    video_path: Path,
    out_dir: Path,
    *,
    max_frames: int | None = None,
    sample_every: int = 2,
    overlay: bool = True,
    overlay_every: int = 10,
    quantize: str = "coarse",
    model_dir: Path | None = None,
    write_video_only: bool = False,
) -> dict[str, Any]:
    """Fuse pyin + landmarks and write MusicXML with fingerings/positions.

    Returns a JSON-serializable summary with absolute paths to artifacts.
    Raises ``SilentAudioError`` / ``FileNotFoundError`` / ``ValueError`` on failure.
    """
    video_path = Path(video_path)
    if video_path.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(
            f"Unsupported format {video_path.suffix!r}; "
            f"use one of: {', '.join(sorted(VIDEO_EXTS))}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    stem = video_path.stem

    result = analyze_video(
        video_path,
        model_dir=models,
        sample_every=sample_every,
        max_frames=max_frames,
        overlay_dir=None,
        overlay_every=overlay_every,
    )
    result = attach_audio_notes(
        result,
        video_path,
        out_dir=out_dir,
        write_splits=True,
        write_video_only=write_video_only,
        require_audible=True,
    )

    vision_json = out_dir / f"{stem}_video_timeline.json"
    write_timeline(result, vision_json)

    fused = fuse_timeline(result)
    for key in ("audio_path", "video_path", "video_only_path", "audio_alignment"):
        if key in result:
            fused[key] = result[key]

    fused_json = out_dir / f"{stem}_fused_timeline.json"
    write_timeline(fused, fused_json)

    sample_path = out_dir / f"{stem}_fused_sample.json"
    sample_doc = {
        "schema": fused.get("schema"),
        "source": fused.get("source"),
        "audio_path": fused.get("audio_path"),
        "video_path": fused.get("video_path"),
        "audio_alignment": fused.get("audio_alignment"),
        "fusion": fused.get("fusion"),
        "fused_notes_preview": (fused.get("fused_notes") or [])[:12],
        "fused_events_preview": (fused.get("fused_events") or [])[:20],
    }
    sample_path.write_text(json.dumps(sample_doc, indent=2), encoding="utf-8")

    overlay_paths: list[str] = []
    if overlay:
        overlay_dir = out_dir / f"{stem}_fused_overlay"
        n_ov = write_fused_overlays(
            video_path,
            fused,
            overlay_dir,
            every=overlay_every,
        )
        if n_ov:
            # Keep a few preview frames for the website.
            jpgs = sorted(overlay_dir.glob("*.jpg"))[:6]
            for i, src in enumerate(jpgs):
                dest = out_dir / f"{stem}_overlay_preview_{i:02d}.jpg"
                shutil.copy2(src, dest)
                overlay_paths.append(str(dest))

    xml_path = out_dir / f"{stem}_fused.musicxml"
    audio_for_score = fused.get("audio_path")
    xml_out, score = write_fused_musicxml(
        fused,
        xml_path,
        audio_path=Path(audio_for_score) if audio_for_score else None,
        quantize=quantize,
        title=stem,
    )
    score_json = out_dir / f"{stem}_fused_score.json"
    write_timeline(score, score_json)

    notes = score.get("notes") or []
    return {
        "ok": True,
        "stem": stem,
        "num_notes": len(notes),
        "num_rests": int(score.get("num_rests") or 0),
        "tempo_bpm": (score.get("tempo") or {}).get("bpm"),
        "key": (score.get("key") or {}).get("name"),
        "musicxml_path": str(xml_out),
        "score_json_path": str(score_json),
        "fused_json_path": str(fused_json),
        "sample_json_path": str(sample_path),
        "audio_path": fused.get("audio_path"),
        "overlay_previews": overlay_paths,
        "max_frames": max_frames,
        "quantize": quantize,
        "source": "fused_transcription",
        "match": None,  # catalog / Shazam-IMSLP paused; always fused MusicXML
        "notes_preview": [
            {
                "note": n.get("note"),
                "midi": n.get("midi"),
                "finger": n.get("score_finger"),
                "position": n.get("score_position"),
                "string": n.get("string"),
            }
            for n in notes[:16]
        ],
        "limits": score.get("limits") or [],
    }


__all__ = ["VIDEO_EXTS", "SilentAudioError", "process_video_to_score"]
