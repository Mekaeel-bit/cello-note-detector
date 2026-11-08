"""Audio extract / track split helpers for video ↔ pyin fusion."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

# Peak-normalized int16 RMS below this ≈ digital silence / broken extract.
DEFAULT_MIN_RMS = 50.0


class SilentAudioError(RuntimeError):
    """Raised when extracted audio has negligible energy."""


def extract_wav(
    video_path: Path,
    wav_path: Path,
    *,
    sr: int = 16000,
) -> Path:
    """Extract mono PCM WAV from any video container via ffmpeg."""
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
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffmpeg not found on PATH (WinError 2). Install ffmpeg and restart "
            "the terminal, e.g. PowerShell: winget install Gyan.FFmpeg"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"ffmpeg extract failed for {video_path}: {stderr}") from exc
    return wav_path


def extract_video_only(video_path: Path, out_path: Path) -> Path:
    """Copy video stream without audio (any input clip)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-c:v",
        "copy",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg before splitting tracks."
        ) from exc
    except subprocess.CalledProcessError as exc:
        # Some containers need re-encode when stream copy fails.
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def wav_rms(wav_path: Path) -> float:
    """Return int16-sample RMS of a mono/stereo WAV file."""
    with wave.open(str(wav_path), "rb") as wf:
        n = wf.getnframes()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        if n <= 0:
            return 0.0
        raw = wf.readframes(n)
    if sw != 2:
        # Fallback via numpy from_buffer only for 16-bit PCM extracts we write.
        return 0.0
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data * data)))


def assert_audible(wav_path: Path, *, min_rms: float = DEFAULT_MIN_RMS) -> float:
    rms = wav_rms(wav_path)
    if rms < min_rms:
        raise SilentAudioError(
            f"Extracted audio looks silent (RMS={rms:.1f} < {min_rms}). "
            f"File: {wav_path}. Re-mux the source with AAC audio or pass a "
            "clip that actually contains sound."
        )
    return rms


def split_av_tracks(
    video_path: Path,
    out_dir: Path,
    *,
    stem: str | None = None,
    sr: int = 16000,
    write_video_only: bool = True,
    min_rms: float = DEFAULT_MIN_RMS,
) -> dict[str, Any]:
    """Write ``<stem>_audio.wav`` (+ optional ``<stem>_video_only.mp4``) beside outputs.

    Works for any local video path (not YouTube-specific). Returns path metadata
    including RMS; raises ``SilentAudioError`` / ``FileNotFoundError`` on failure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = stem or Path(video_path).stem
    audio_path = out_dir / f"{name}_audio.wav"
    video_only_path = out_dir / f"{name}_video_only.mp4"

    extract_wav(video_path, audio_path, sr=sr)
    rms = assert_audible(audio_path, min_rms=min_rms)

    meta: dict[str, Any] = {
        "audio_path": str(audio_path),
        "video_path": str(video_path),
        "sample_rate": sr,
        "rms": round(rms, 2),
        "min_rms": min_rms,
    }
    if write_video_only:
        extract_video_only(video_path, video_only_path)
        meta["video_only_path"] = str(video_only_path)
    return meta


def attach_audio_notes(
    timeline: dict[str, Any],
    video_path: Path,
    *,
    out_dir: Path | None = None,
    write_splits: bool = False,
    write_video_only: bool = True,
    min_rms: float = DEFAULT_MIN_RMS,
    require_audible: bool = False,
) -> dict[str, Any]:
    """Best-effort: add ``audio_notes`` from the existing pyin pipeline.

    When ``write_splits`` is True (``--fuse``), persist WAV (+ optional video-only
    MP4) under ``out_dir`` and record ``audio_path`` / ``video_path`` on the
    timeline. Silent extracts set ``audio_alignment.ok=False``; if
    ``require_audible`` is True they also raise ``SilentAudioError``.
    """
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

    video_path = Path(video_path)
    split_meta: dict[str, Any] | None = None
    wav_for_pyin: Path

    try:
        if write_splits:
            if out_dir is None:
                raise ValueError("out_dir is required when write_splits=True")
            split_meta = split_av_tracks(
                video_path,
                Path(out_dir),
                stem=video_path.stem,
                write_video_only=write_video_only,
                min_rms=min_rms,
            )
            wav_for_pyin = Path(split_meta["audio_path"])
        else:
            # Ephemeral extract (align-audio without persisting splits).
            import tempfile

            tmp = tempfile.mkdtemp(prefix="cnd_av_")
            wav_for_pyin = Path(tmp) / "audio.wav"
            extract_wav(video_path, wav_for_pyin)
            rms = assert_audible(wav_for_pyin, min_rms=min_rms)
            split_meta = {
                "audio_path": str(wav_for_pyin),
                "video_path": str(video_path),
                "rms": round(rms, 2),
                "ephemeral": True,
            }
    except SilentAudioError as exc:
        timeline["audio_alignment"] = {
            "ok": False,
            "error": str(exc),
            "video_path": str(video_path),
            **({k: v for k, v in (split_meta or {}).items() if k.endswith("_path")}),
        }
        if require_audible:
            raise
        return timeline
    except FileNotFoundError as exc:
        timeline["audio_alignment"] = {
            "ok": False,
            "error": str(exc),
            "video_path": str(video_path),
        }
        if require_audible:
            raise
        return timeline
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        timeline["audio_alignment"] = {
            "ok": False,
            "error": f"ffmpeg extract failed: {exc}",
            "video_path": str(video_path),
        }
        if require_audible:
            raise
        return timeline

    y, sr = load_mono(wav_for_pyin)
    notes, _meta = transcribe_notes(y, sr=sr)
    note_dicts = note_events_to_dicts(notes)

    timeline["audio_path"] = split_meta.get("audio_path")
    timeline["video_path"] = split_meta.get("video_path", str(video_path))
    if split_meta.get("video_only_path"):
        timeline["video_only_path"] = split_meta["video_only_path"]

    timeline["audio_alignment"] = {
        "ok": True,
        "method": "ffmpeg-extract + pyin",
        "num_notes": len(note_dicts),
        "note": "Rough sync by shared clock; no AV drift correction.",
        "rms": split_meta.get("rms"),
        "audio_path": timeline["audio_path"],
        "video_path": timeline["video_path"],
        **(
            {"video_only_path": timeline["video_only_path"]}
            if "video_only_path" in timeline
            else {}
        ),
    }
    timeline["audio_notes"] = note_dicts

    # Cross-link nearest bow/finger events to each note onset (±0.35s)
    events = timeline.get("events") or []
    for n in note_dicts:
        onset = float(n.get("onset", 0.0))
        nearby = [e for e in events if abs(float(e["t"]) - onset) <= 0.35]
        n["nearby_video_events"] = nearby
    return timeline
