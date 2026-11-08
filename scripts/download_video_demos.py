#!/usr/bin/env python3
"""Fetch bundled front-facing cello demo clips + MediaPipe models.

YouTube is often bot-gated in cloud VMs; we pull Internet Archive mirrors
of the same watch URLs (see data/video/SOURCES.md).
"""

from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "video" / "clips"
RAW = ROOT / "data" / "video" / "raw"
MODELS = ROOT / "data" / "video" / "models"

# identifier, filename on IA, local stem, trim start/duration for demo clip
DEMOS = [
    {
        "yt": "https://www.youtube.com/watch?v=KZ4Id52zQw8",
        "ia": "youtube-KZ4Id52zQw8",
        "ia_file": "KZ4Id52zQw8.mp4",
        "clip": "wmea_jr_scale_etude_front.mp4",
        "ss": 8,
        "t": 20,
    },
    {
        "yt": "https://www.youtube.com/watch?v=_jecuyxPO6s",
        "ia": "youtube-_jecuyxPO6s",
        "ia_file": "_jecuyxPO6s.mp4",
        "clip": "wmea_hs_chromatic_scale_front.mp4",
        "ss": 8,
        "t": 18,
    },
    {
        "yt": "https://www.youtube.com/watch?v=stMH0JKsqRQ",
        "ia": "youtube-stMH0JKsqRQ",
        "ia_file": "stMH0JKsqRQ.mp4",
        "clip": "wmea_jr_scale_etude_intro_front.mp4",
        "ss": 35,
        "t": 25,
    },
    {
        "yt": "https://www.youtube.com/watch?v=yNxSElVqKcM",
        "ia": "youtube-yNxSElVqKcM",
        "ia_file": "yNxSElVqKcM.mp4",
        "clip": "wmea_hs_chromatic_intro_front.mp4",
        "ss": 30,
        "t": 25,
    },
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {url}\n → {dest}")
    urllib.request.urlretrieve(url, dest)


def download_models() -> None:
    from cello_note_detector.video.models import ensure_models

    ensure_models(MODELS)


def download_clips(*, force: bool = False, keep_raw: bool = False) -> None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for demo in DEMOS:
        clip_path = CLIPS / demo["clip"]
        if clip_path.is_file() and clip_path.stat().st_size > 50_000 and not force:
            print(f"skip existing {clip_path.name}")
            continue
        raw_path = RAW / demo["ia_file"]
        url = f"https://archive.org/download/{demo['ia']}/{demo['ia_file']}"
        if not raw_path.is_file() or force:
            _download(url, raw_path)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(demo["ss"]),
            "-i",
            str(raw_path),
            "-t",
            str(demo["t"]),
            "-vf",
            "scale='min(720,iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            str(clip_path),
        ]
        print(" ", " ".join(cmd[-6:]))
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"wrote {clip_path} (from {demo['yt']})")
        if not keep_raw:
            raw_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-only", action="store_true")
    ap.add_argument("--clips-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()
    if args.models_only:
        download_models()
        return
    if args.clips_only:
        download_clips(force=args.force, keep_raw=args.keep_raw)
        return
    download_clips(force=args.force, keep_raw=args.keep_raw)
    download_models()


if __name__ == "__main__":
    main()
