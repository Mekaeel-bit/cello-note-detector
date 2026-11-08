"""Tests for track split + silent-audio detection."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from cello_note_detector.video.align_audio import (
    SilentAudioError,
    assert_audible,
    split_av_tracks,
    wav_rms,
)


def _write_wav(path: Path, *, samples: list[int], sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def test_wav_rms_silent_and_loud(tmp_path: Path):
    silent = tmp_path / "silent.wav"
    loud = tmp_path / "loud.wav"
    _write_wav(silent, samples=[0] * 1600)
    _write_wav(loud, samples=[8000, -8000] * 800)
    assert wav_rms(silent) < 1.0
    assert wav_rms(loud) > 1000.0
    with pytest.raises(SilentAudioError):
        assert_audible(silent)
    assert assert_audible(loud) > 1000.0


def test_split_av_tracks_writes_paths(tmp_path: Path):
    # Tiny synthetic "video" via ffmpeg color+sine if available.
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    src = tmp_path / "tone.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.5",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=0.5",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    out = tmp_path / "out"
    meta = split_av_tracks(src, out, stem="tone")
    assert Path(meta["audio_path"]).is_file()
    assert Path(meta["video_only_path"]).is_file()
    assert meta["rms"] >= 50.0
    assert meta["video_path"] == str(src)
