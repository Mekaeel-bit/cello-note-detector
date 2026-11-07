#!/usr/bin/env python3
"""Generate short synthetic monophonic cello-range phrases for multi-note tests.

Renders additive (harmonic) tones — not a sampled cello — so clips stay
license-clean. Expected MIDI pitches are written beside each WAV as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44_100
# Soft saw-like partials (odd+even), decaying amplitude ≈ bowed string.
HARMONIC_GAINS = [1.0, 0.55, 0.35, 0.22, 0.14, 0.09, 0.06, 0.04]


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def synth_note(
    midi: int,
    duration: float,
    *,
    sr: int = SR,
    amplitude: float = 0.28,
    attack: float = 0.02,
    release: float = 0.04,
) -> np.ndarray:
    n = max(1, int(round(duration * sr)))
    t = np.arange(n, dtype=np.float64) / sr
    freq = midi_to_hz(midi)
    wave = np.zeros(n, dtype=np.float64)
    for k, gain in enumerate(HARMONIC_GAINS, start=1):
        partial = freq * k
        if partial >= sr * 0.45:
            break
        wave += gain * np.sin(2.0 * math.pi * partial * t)

    # Mild amplitude vibrato / bow noise proxy.
    wave *= 1.0 + 0.015 * np.sin(2.0 * math.pi * 5.5 * t)
    env = np.ones(n, dtype=np.float64)
    a = min(int(attack * sr), n // 3)
    r = min(int(release * sr), n // 3)
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    if r > 0:
        env[-r:] = np.linspace(1.0, 0.0, r, endpoint=True)
    wave *= env
    peak = np.max(np.abs(wave)) or 1.0
    return (amplitude * wave / peak).astype(np.float32)


def render_phrase(
    notes: list[tuple[int, float]],
    *,
    gap: float = 0.04,
    sr: int = SR,
) -> tuple[np.ndarray, list[dict]]:
    """Render sequential notes with short gaps. Returns audio + expected events."""
    parts: list[np.ndarray] = []
    expected: list[dict] = []
    t = 0.0
    silence = np.zeros(max(1, int(round(gap * sr))), dtype=np.float32)

    for i, (midi, dur) in enumerate(notes):
        note_audio = synth_note(midi, dur, sr=sr)
        parts.append(note_audio)
        expected.append(
            {
                "midi": midi,
                "onset": round(t, 4),
                "offset": round(t + dur, 4),
                "duration": round(dur, 4),
            }
        )
        t += dur
        if i < len(notes) - 1 and gap > 0:
            parts.append(silence)
            t += gap

    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    # Tiny leading/trailing pad so onset detectors see a clean attack.
    pad = np.zeros(int(0.05 * sr), dtype=np.float32)
    audio = np.concatenate([pad, audio, pad])
    shift = 0.05
    for e in expected:
        e["onset"] = round(e["onset"] + shift, 4)
        e["offset"] = round(e["offset"] + shift, 4)
    return audio, expected


PHRASES: dict[str, dict] = {
    "scale_c_major_cello": {
        "description": (
            "Ascending C-major scale fragment in cello tenor range: "
            "C3 D3 E3 F3 G3 A3 B3 C4 (quarter notes @ ~100 BPM)."
        ),
        "notes": [
            (48, 0.55),
            (50, 0.55),
            (52, 0.55),
            (53, 0.55),
            (55, 0.55),
            (57, 0.55),
            (59, 0.55),
            (60, 0.70),
        ],
        "gap": 0.05,
    },
    "twinkle_opening": {
        "description": (
            "Opening of Twinkle Twinkle (public-domain melody) transposed "
            "for cello: C3 C3 G3 G3 A3 A3 G3."
        ),
        "notes": [
            (48, 0.45),
            (48, 0.45),
            (55, 0.45),
            (55, 0.45),
            (57, 0.45),
            (57, 0.45),
            (55, 0.80),
        ],
        "gap": 0.06,
    },
    "open_strings_arpeggio": {
        "description": (
            "Cello open-string style arpeggio: C2 G2 D3 A3 then descending "
            "A3 D3 G2 C2 (synthetic harmonics, not a real open-string sample)."
        ),
        "notes": [
            (36, 0.50),
            (43, 0.50),
            (50, 0.50),
            (57, 0.50),
            (57, 0.40),
            (50, 0.40),
            (43, 0.40),
            (36, 0.60),
        ],
        "gap": 0.05,
    },
    "stepwise_phrase_g": {
        "description": (
            "Short stepwise G-minor-ish phrase: G2 Bb2 D3 F3 Eb3 D3 C3 Bb2 G2."
        ),
        "notes": [
            (43, 0.40),
            (46, 0.40),
            (50, 0.40),
            (53, 0.40),
            (51, 0.40),
            (50, 0.40),
            (48, 0.40),
            (46, 0.40),
            (43, 0.65),
        ],
        "gap": 0.045,
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "multi_note_phrases",
        help="Directory for WAV + expected-note JSON sidecars",
    )
    args = parser.parse_args(argv)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    for name, spec in PHRASES.items():
        audio, expected = render_phrase(spec["notes"], gap=spec["gap"], sr=SR)
        wav_path = out_dir / f"{name}.wav"
        meta_path = out_dir / f"{name}_expected.json"
        sf.write(str(wav_path), audio, SR, subtype="PCM_16")
        meta = {
            "name": name,
            "description": spec["description"],
            "synthesis": "additive harmonic (cello-range), not a sampled instrument",
            "sample_rate": SR,
            "duration_sec": round(len(audio) / SR, 4),
            "num_notes": len(expected),
            "expected_notes": expected,
            "source": "scripts/generate_multi_note_phrases.py",
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        index.append(
            {
                "file": wav_path.name,
                "expected": meta_path.name,
                "num_notes": meta["num_notes"],
                "description": spec["description"],
            }
        )
        print(f"wrote {wav_path} ({meta['num_notes']} notes, {meta['duration_sec']}s)")

    (out_dir / "README.md").write_text(
        _readme(index),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'README.md'}")
    return 0


def _readme(index: list[dict]) -> str:
    lines = [
        "# Multi-note test phrases",
        "",
        "Synthetic **monophonic** cello-range melodies for multi-note transcription tests.",
        "Generated by `scripts/generate_multi_note_phrases.py` (additive harmonics).",
        "These are **not** copyrighted solo recordings.",
        "",
        "| File | Notes | Description |",
        "|------|------:|-------------|",
    ]
    for row in index:
        desc = row["description"].replace("|", "/")
        lines.append(
            f"| `{row['file']}` | {row['num_notes']} | {desc} |"
        )
    lines.extend(
        [
            "",
            "Each WAV has a sibling `*_expected.json` with ground-truth MIDI onsets/offsets",
            "from the renderer (ideal timing; pyin output will differ slightly).",
            "",
            "Regenerate:",
            "",
            "```bash",
            "python scripts/generate_multi_note_phrases.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
