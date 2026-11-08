"""python -m cello_note_detector.video <video> -o outputs/video [--fuse]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cello_note_detector.video.align_audio import attach_audio_notes
from cello_note_detector.video.detect import analyze_video, write_timeline
from cello_note_detector.video.fusion import fuse_timeline
from cello_note_detector.video.models import DEFAULT_MODEL_DIR
from cello_note_detector.video.overlay_fuse import write_fused_overlays

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLIPS = _REPO_ROOT / "data" / "video" / "clips"
DEFAULT_OUT = _REPO_ROOT / "outputs" / "video"


def _discover_videos(path: Path) -> list[Path]:
    exts = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
    if path.is_file():
        if path.suffix.lower() not in exts:
            raise SystemExit(f"Expected a video file, got: {path}")
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Path not found: {path}")
    vids = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
    if not vids:
        raise SystemExit(f"No video files found under: {path}")
    return vids


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Sandbox: fingering/bow timeline from front-facing cello video. "
            "Use --fuse to combine pyin pitch with landmarks for string-aware labels."
        )
    )
    p.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_CLIPS),
        help="Video file or directory (default: data/video/clips)",
    )
    p.add_argument("-o", "--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--sample-every", type=int, default=2, help="Process every Nth frame")
    p.add_argument("--max-frames", type=int, default=None, help="Cap processed frames")
    p.add_argument("--overlay", action="store_true", help="Write annotated JPEG frames")
    p.add_argument("--overlay-every", type=int, default=10)
    p.add_argument(
        "--align-audio",
        action="store_true",
        help="Extract audio + pyin notes (cross-link only; see also --fuse)",
    )
    p.add_argument(
        "--fuse",
        action="store_true",
        help=(
            "Fuse pyin pitch with hand landmarks → string-aware fingering + bow string "
            "(implies audio extract; writes *_fused_timeline.json)"
        ),
    )
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    args = p.parse_args(argv)

    videos = _discover_videos(Path(args.path))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for video in videos:
        print(f"Analyzing {video} …")
        # When fusing, draw fused overlays in a post-pass (richer HUD).
        vision_overlay = (
            out_dir / f"{video.stem}_overlay" if args.overlay and not args.fuse else None
        )
        result = analyze_video(
            video,
            model_dir=Path(args.model_dir),
            sample_every=args.sample_every,
            max_frames=args.max_frames,
            overlay_dir=vision_overlay,
            overlay_every=args.overlay_every,
        )
        if args.fuse or args.align_audio:
            result = attach_audio_notes(result, video)

        out_json = out_dir / f"{video.stem}_video_timeline.json"
        write_timeline(result, out_json)
        n_ev = len(result.get("events") or [])
        print(
            f"  frames={result['frames_sampled']} events={n_ev} → {out_json}"
            + (f" overlay={vision_overlay}" if vision_overlay else "")
        )

        if args.fuse:
            fused = fuse_timeline(result)
            fused_json = out_dir / f"{video.stem}_fused_timeline.json"
            write_timeline(fused, fused_json)
            # Compact sample for quick inspection (first few fused notes).
            sample_path = out_dir / f"{video.stem}_fused_sample.json"
            sample_doc = {
                "schema": fused.get("schema"),
                "source": fused.get("source"),
                "fusion": fused.get("fusion"),
                "fused_notes_preview": (fused.get("fused_notes") or [])[:12],
                "fused_events_preview": (fused.get("fused_events") or [])[:20],
            }
            sample_path.write_text(json.dumps(sample_doc, indent=2), encoding="utf-8")
            n_fn = len(fused.get("fused_notes") or [])
            n_fe = len(fused.get("fused_events") or [])
            msg = f"  fused notes={n_fn} events={n_fe} → {fused_json} sample={sample_path}"
            if args.overlay:
                overlay_dir = out_dir / f"{video.stem}_fused_overlay"
                n_ov = write_fused_overlays(
                    video,
                    fused,
                    overlay_dir,
                    every=args.overlay_every,
                )
                msg += f" overlay={overlay_dir} ({n_ov} frames)"
            print(msg)


if __name__ == "__main__":
    main()
