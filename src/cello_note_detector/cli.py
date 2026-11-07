"""CLI: pyin note events + MERT frame embeddings for cello WAVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cello_note_detector.audio.load import TARGET_SR, load_mono
from cello_note_detector.audio.mert import (
    DEFAULT_MODEL_ID,
    extract_mert_embeddings,
    load_mert,
    summarize_embeddings,
)
from cello_note_detector.audio.pitch_baseline import detect_f0, f0_to_note_events
from cello_note_detector.decode.events import note_events_to_dicts

# Repo root: .../src/cello_note_detector/cli.py → parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDINGS = _REPO_ROOT / "data" / "Recordings"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs"


def _discover_wavs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".wav":
            raise SystemExit(f"Expected a .wav file, got: {path}")
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Path not found: {path}")
    wavs = sorted(path.glob("*.wav")) + sorted(path.glob("*.WAV"))
    # de-dupe while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for w in wavs:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    if not unique:
        raise SystemExit(f"No WAV files found under: {path}")
    return unique


def _process_one(
    wav_path: Path,
    *,
    output_dir: Path,
    mert_bundle,
    save_embeddings: bool,
    skip_mert: bool,
    min_duration: float,
) -> dict:
    audio, sr = load_mono(wav_path, sr=TARGET_SR)
    duration_s = float(len(audio) / sr)

    times, f0, voiced_flag, voiced_probs = detect_f0(audio, sr=sr)
    events = f0_to_note_events(
        times, f0, voiced_flag, voiced_probs, sr=sr, min_duration=min_duration
    )
    note_dicts = note_events_to_dicts(events)

    mert_info: dict | None = None
    emb_path: str | None = None
    if not skip_mert:
        embeddings = extract_mert_embeddings(audio, mert_bundle, sr=sr)
        mert_info = {
            "model_id": mert_bundle.model_id,
            "device": str(mert_bundle.device),
            "summary": summarize_embeddings(embeddings),
        }
        if save_embeddings:
            emb_file = output_dir / f"{wav_path.stem}_mert_embeddings.npy"
            np.save(emb_file, embeddings)
            emb_path = str(emb_file)
            mert_info["embeddings_path"] = emb_path

    result = {
        "audio_path": str(wav_path),
        "sample_rate": sr,
        "duration_sec": round(duration_s, 4),
        "pitch_method": "librosa.pyin",
        "cello_range": {"fmin": "C2", "fmax": "A5"},
        "num_notes": len(note_dicts),
        "notes": note_dicts,
        "mert": mert_info,
    }

    json_path = output_dir / f"{wav_path.stem}_notes.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["_json_path"] = str(json_path)
    return result


def _print_summary(result: dict) -> None:
    print(f"\n=== {Path(result['audio_path']).name} ===")
    print(
        f"duration={result['duration_sec']:.2f}s  "
        f"notes={result['num_notes']}  "
        f"json={result['_json_path']}"
    )
    preview = result["notes"][:8]
    for n in preview:
        print(
            f"  {n['onset']:6.2f}–{n['offset']:6.2f}s  "
            f"{n['note']:>4}  midi={n['midi']:3d}  "
            f"conf={n['confidence']:.2f}"
        )
    if result["num_notes"] > len(preview):
        print(f"  … {result['num_notes'] - len(preview)} more notes")

    mert = result.get("mert")
    if mert:
        shape = mert["summary"]["shape"]
        print(
            f"  MERT {mert['model_id']} on {mert['device']}: "
            f"embeddings shape={shape}"
        )
        if "embeddings_path" in mert:
            print(f"  saved embeddings → {mert['embeddings_path']}")
    else:
        print("  MERT: skipped")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cello-note-detector",
        description=(
            "Milestone 1: monophonic cello WAV → pyin note events + "
            "MERT-v1-95M frame embeddings."
        ),
    )
    p.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_RECORDINGS),
        help=(
            "WAV file or directory of WAVs "
            f"(default: {DEFAULT_RECORDINGS})"
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON (and optional .npy) outputs",
    )
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face MERT model id (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Also write full MERT embeddings as .npy next to the JSON",
    )
    p.add_argument(
        "--skip-mert",
        action="store_true",
        help="Only run pyin note detection (no model download)",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=0.05,
        help="Drop note events shorter than this many seconds (default: 0.05)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wavs = _discover_wavs(Path(args.path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mert_bundle = None
    if not args.skip_mert:
        print(f"Loading {args.model_id} …")
        mert_bundle = load_mert(args.model_id, device=args.device)
        print(f"MERT ready on {mert_bundle.device} (hidden={mert_bundle.hidden_size})")

    print(f"Processing {len(wavs)} file(s) → {output_dir}")
    for wav in wavs:
        result = _process_one(
            wav,
            output_dir=output_dir,
            mert_bundle=mert_bundle,
            save_embeddings=args.save_embeddings,
            skip_mert=args.skip_mert,
            min_duration=args.min_duration,
        )
        _print_summary(result)

    print(f"\nDone. Wrote {len(wavs)} JSON file(s) under {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
