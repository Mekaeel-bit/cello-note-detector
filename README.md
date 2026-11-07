# cello-note-detector

Hackathon app for identifying cello notes from **monophonic audio** (video / fingerings later).

**Current milestone:** multi-note monophonic melody transcription via a tuned **pyin** baseline + onset-aware segmentation, plus optional **MERT-v1-95M** frame embeddings. No trained pitch head, rhythm quantization, or sheet-music export yet.

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

## Run

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
| `--no-onsets` | disable onset-based note splits |
| `--gap-tolerance 0.06` | bridge short unvoiced gaps inside one note |
| `--min-duration 0.04` | drop flicker shorter than this |

### Output

For each WAV, writes `outputs/<stem>_notes.json` with:

- `notes`: list of `{pitch_hz, midi, note, onset, offset, duration, confidence}` from pyin
- `segmentation`: hop/frame settings, onset times, gap/median options
- `num_notes`: event count
- `mert.summary`: embedding shape / mean / std (and optional `embeddings_path` if `--save-embeddings`)

Console prints a short note preview plus MERT shape.

## Multi-note tests

The sample-pack loops under `data/Recordings/` are mostly sustained single pitches. For **multi-note** melody checks, use the synthetic phrases in `data/multi_note_phrases/` (additive cello-range tones — license-clean; see that folder’s README).

```bash
# pyin-only multi-note transcription (fast)
python -m cello_note_detector data/multi_note_phrases/ -o outputs/multi_note --skip-mert

# one phrase
python -m cello_note_detector data/multi_note_phrases/scale_c_major_cello.wav -o outputs/multi_note --skip-mert
```

Expect `num_notes` > 1 (e.g. 8 for the C-major scale, 7 for Twinkle opening). Sibling `*_expected.json` files list the renderer’s ground-truth MIDI for comparison.

Regenerate phrases after editing the generator:

```bash
python scripts/generate_multi_note_phrases.py
```

## Layout

```text
src/cello_note_detector/
  audio/          # load, pyin + onsets, MERT embed
  decode/         # note event types
  cli.py          # entrypoint
scripts/
  generate_multi_note_phrases.py
data/Recordings/          # sample-pack cello WAVs
data/multi_note_phrases/  # synthetic multi-note test clips
outputs/                  # generated JSON / .npy (gitignored)
```

## Notes / limits

- Scope v1: **audio-only**, monophonic cello, **MERT-v1-95M** (not 330M).
- Transformers is pinned to **4.43.4** for MERT’s custom code.
- Segmentation splits on MIDI pitch changes, short gaps, and spectral onsets (so repeated same-pitch notes can separate). It is still a classical baseline — vibrato, slides, and noisy real recordings can merge/split notes incorrectly.
- Out of scope for this milestone: rhythm quantization, MusicXML/sheet export, video, MERT fine-tuning.
