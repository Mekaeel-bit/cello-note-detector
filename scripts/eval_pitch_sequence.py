#!/usr/bin/env python3
"""Compare a transcription JSON pitch sequence to a monophonic reference.

Used by the Bach Prelude accuracy sandbox (Mutopia/Schirmer = IMSLP-aligned
public-domain pitch/rhythm ground truth). Metrics are piece-agnostic.

Example:
  python -m cello_note_detector data/multi_note_phrases/Bach_Prelude_from_Cello_Suite_in_G.wav \\
      -o outputs/bach_sandbox --skip-mert --quantize score --musicxml --midi
  python scripts/eval_pitch_sequence.py \\
      outputs/bach_sandbox/Bach_Prelude_from_Cello_Suite_in_G_notes.json \\
      data/reference/bwv1007_prelude/prelude_monophonic_reference.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_hyp_midi(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    notes = data.get("notes") or []
    return [int(n["midi"]) for n in notes]


def _load_ref(path: Path) -> tuple[list[int], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    seq = data.get("midi_sequence")
    if seq is None and "events" in data:
        seq = [int(e["midi"]) for e in data["events"]]
    if seq is None:
        raise SystemExit(f"No midi_sequence/events in reference: {path}")
    return [int(x) for x in seq], data


def _levenshtein(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _best_window_accuracy(hyp: list[int], ref: list[int]) -> dict:
    """Slide hyp over ref (or vice versa) and report best exact-match rate."""
    if not hyp or not ref:
        return {
            "pitch_accuracy": 0.0,
            "matched": 0,
            "compared": 0,
            "offset": 0,
            "mode": "empty",
        }

    # Case 1: hyp is a contiguous excerpt of ref (short test WAV).
    best = {"pitch_accuracy": -1.0, "matched": 0, "compared": 0, "offset": 0, "mode": ""}
    if len(hyp) <= len(ref):
        for off in range(0, len(ref) - len(hyp) + 1):
            window = ref[off : off + len(hyp)]
            matched = sum(1 for h, r in zip(hyp, window) if h == r)
            acc = matched / len(hyp)
            if acc > best["pitch_accuracy"]:
                best = {
                    "pitch_accuracy": acc,
                    "matched": matched,
                    "compared": len(hyp),
                    "offset": off,
                    "mode": "hyp_in_ref",
                }
    # Case 2: ref excerpt inside hyp (over-segmented / longer hyp).
    if len(ref) <= len(hyp):
        for off in range(0, len(hyp) - len(ref) + 1):
            window = hyp[off : off + len(ref)]
            matched = sum(1 for h, r in zip(window, ref) if h == r)
            acc = matched / len(ref)
            if acc > best["pitch_accuracy"]:
                best = {
                    "pitch_accuracy": acc,
                    "matched": matched,
                    "compared": len(ref),
                    "offset": off,
                    "mode": "ref_in_hyp",
                }
    return best


def _duration_stats(hyp_path: Path) -> dict:
    data = json.loads(hyp_path.read_text(encoding="utf-8"))
    notes = data.get("notes") or []
    vals = [n.get("duration_value") for n in notes if n.get("duration_value")]
    from collections import Counter

    counts = Counter(vals)
    n = max(len(vals), 1)
    sixteenth_frac = counts.get("sixteenth", 0) / n
    return {
        "num_notes": len(notes),
        "num_rests": data.get("num_rests"),
        "bpm": (data.get("tempo") or {}).get("bpm"),
        "key": (data.get("key") or {}).get("name"),
        "duration_value_counts": dict(counts),
        "sixteenth_fraction": round(sixteenth_frac, 4),
        "equal_stream": (data.get("rhythm") or {}).get("equal_stream"),
        "quantize": (data.get("rhythm") or {}).get("quantize"),
    }


def evaluate(hyp_path: Path, ref_path: Path, *, max_ref_notes: int | None = None) -> dict:
    hyp = _load_hyp_midi(hyp_path)
    ref, ref_meta = _load_ref(ref_path)
    if max_ref_notes is None and ref_meta.get("verified_prefix_notes"):
        # Prefer the hand-checked opening when scoring short clips.
        max_ref_notes = int(ref_meta["verified_prefix_notes"])
    if max_ref_notes is not None:
        ref = ref[: max_ref_notes]
    window = _best_window_accuracy(hyp, ref)
    dist = _levenshtein(hyp, ref[: max(len(hyp), 1)] if window["mode"] == "hyp_in_ref" else ref)
    # Normalized edit similarity on the compared window.
    if window["mode"] == "hyp_in_ref":
        ref_win = ref[window["offset"] : window["offset"] + len(hyp)]
        edit = _levenshtein(hyp, ref_win)
        norm = 1.0 - edit / max(len(hyp), 1)
    elif window["mode"] == "ref_in_hyp":
        hyp_win = hyp[window["offset"] : window["offset"] + len(ref)]
        edit = _levenshtein(hyp_win, ref)
        norm = 1.0 - edit / max(len(ref), 1)
    else:
        edit = dist
        norm = 0.0

    rhythm = _duration_stats(hyp_path)
    return {
        "hyp": str(hyp_path),
        "ref": str(ref_path),
        "ref_edition": ref_meta.get("source_edition"),
        "hyp_notes": len(hyp),
        "ref_notes_used": len(ref),
        "pitch_accuracy": round(float(window["pitch_accuracy"]), 4),
        "pitch_matched": window["matched"],
        "pitch_compared": window["compared"],
        "alignment_mode": window["mode"],
        "alignment_offset": window["offset"],
        "edit_distance": edit,
        "sequence_similarity": round(float(norm), 4),
        "hyp_preview": hyp[:24],
        "ref_preview": (
            ref[window["offset"] : window["offset"] + 24]
            if window["mode"] == "hyp_in_ref"
            else ref[:24]
        ),
        "rhythm": rhythm,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hyp_json", type=Path, help="Transcription *_notes.json")
    p.add_argument(
        "ref_json",
        type=Path,
        nargs="?",
        default=Path("data/reference/bwv1007_prelude/prelude_monophonic_reference.json"),
        help="Reference JSON with midi_sequence",
    )
    p.add_argument(
        "--max-ref-notes",
        type=int,
        default=None,
        help="Only score against the first N reference notes (short clips)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the metrics JSON",
    )
    args = p.parse_args(argv)
    result = evaluate(args.hyp_json, args.ref_json, max_ref_notes=args.max_ref_notes)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
