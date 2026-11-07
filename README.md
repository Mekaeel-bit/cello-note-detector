# cello-note-detector

Hackathon app for identifying cello notes from **monophonic audio** (video / fingerings later).

**Milestone 1:** CLI that runs a classical **pyin** pitch baseline for note events and extracts **MERT-v1-95M** frame embeddings on the same WAV. No trained pitch head yet.

## Setup

Python **3.11+** recommended.

```bash
cd cello-note-detector
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) if you need a CUDA-specific torch build.

## Run (Milestone 1)

Process every WAV under `data/Recordings/` (note the capital **R**):

```bash
python -m cello_note_detector
```

Or a single file / custom directory:

```bash
python -m cello_note_detector data/Recordings/Am_LFCello_SP_01_376.wav
python -m cello_note_detector data/Recordings/ -o outputs --save-embeddings
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--skip-mert` | pyin notes only (no HF model download) |
| `--save-embeddings` | also write `[T, 768]` `.npy` next to the JSON |
| `--device cpu` | force CPU |
| `-o outputs` | output directory (default: `outputs/`) |

### Output

For each WAV, writes `outputs/<stem>_notes.json` with:

- `notes`: list of `{pitch_hz, midi, note, onset, offset, duration, confidence}` from pyin
- `mert.summary`: embedding shape / mean / std (and optional `embeddings_path` if `--save-embeddings`)

Console prints a short note preview plus MERT shape.

## Layout

```text
src/cello_note_detector/
  audio/          # load, pyin baseline, MERT embed
  decode/         # note event types
  cli.py          # entrypoint
data/Recordings/  # sample cello WAVs
outputs/          # generated JSON / .npy (gitignored)
```

## Notes

- Scope v1: **audio-only**, monophonic cello, **MERT-v1-95M** (not 330M).
- Transformers is pinned to **4.43.4** for MERT’s custom code.
- Old one-off scripts (`test.py`, `test_pitch.py`) are replaced by this CLI.
