"""python -m cello_note_detector.video <video> -o outputs/video [--fuse]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cello_note_detector.video.align_audio import SilentAudioError, attach_audio_notes
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
            "(writes *_audio.wav, *_video_only.mp4, *_fused_timeline.json)"
        ),
    )
    p.add_argument(
        "--no-video-only",
        action="store_true",
        help="With --fuse, skip writing <stem>_video_only.mp4 (still writes *_audio.wav)",
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
            try:
                result = attach_audio_notes(
                    result,
                    video,
                    out_dir=out_dir if args.fuse else None,
                    write_splits=bool(args.fuse),
                    write_video_only=bool(args.fuse) and not args.no_video_only,
                    require_audible=bool(args.fuse),
                )
            except SilentAudioError as exc:
                raise SystemExit(f"ERROR: {exc}") from exc
            except FileNotFoundError as exc:
                raise SystemExit(f"ERROR: {exc}") from exc

        out_json = out_dir / f"{video.stem}_video_timeline.json"
        write_timeline(result, out_json)
        n_ev = len(result.get("events") or [])
        aa = result.get("audio_alignment") or {}
        split_msg = ""
        if aa.get("audio_path"):
            split_msg = f" audio={aa['audio_path']}"
            if aa.get("video_only_path"):
                split_msg += f" video_only={aa['video_only_path']}"
            if aa.get("rms") is not None:
                split_msg += f" rms={aa['rms']}"
        print(
            f"  frames={result['frames_sampled']} events={n_ev} → {out_json}"
            + (f" overlay={vision_overlay}" if vision_overlay else "")
            + split_msg
        )

        if args.fuse:
            fused = fuse_timeline(result)
            # Carry explicit split paths onto the fused document.
            for key in ("audio_path", "video_path", "video_only_path", "audio_alignment"):
                if key in result:
                    fused[key] = result[key]
            fused_json = out_dir / f"{video.stem}_fused_timeline.json"
            write_timeline(fused, fused_json)
            sample_path = out_dir / f"{video.stem}_fused_sample.json"
            sample_doc = {
                "schema": fused.get("schema"),
                "source": fused.get("source"),
                "audio_path": fused.get("audio_path"),
                "video_path": fused.get("video_path"),
                "video_only_path": fused.get("video_only_path"),
                "audio_alignment": fused.get("audio_alignment"),
                "fusion": fused.get("fusion"),
                "fused_notes_preview": (fused.get("fused_notes") or [])[:12],
                "fused_events_preview": (fused.get("fused_events") or [])[:20],
            }
            sample_path.write_text(json.dumps(sample_doc, indent=2), encoding="utf-8")
            n_fn = len(fused.get("fused_notes") or [])
            n_fe = len(fused.get("fused_events") or [])
            msg = (
                f"  fused notes={n_fn} events={n_fe} → {fused_json} sample={sample_path}"
                f" audio={fused.get('audio_path')} video={fused.get('video_path')}"
            )
            if fused.get("video_only_path"):
                msg += f" video_only={fused['video_only_path']}"
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
