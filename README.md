# cello-note-detector

Hackathon app for identifying cello notes from **monophonic audio** (video / fingerings later).

**Current milestone:** multi-note monophonic melody transcription (pyin + onsets) → **tempo / beat grid + quantized note durations and rests**, plus optional **MERT-v1-95M** frame embeddings. Not full sheet engraving yet.

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

### Rhythm analysis (tempo + quantization + rests)

Rhythm runs **by default** after note detection (classical DSP: `librosa.beat.beat_track` blended with note-onset IOIs). No MERT download needed:

```bash
# Multi-note phrases (recommended for rhythm checks)
python -m cello_note_detector data/multi_note_phrases/ -o outputs/rhythm --skip-mert

# Dedicated rest clip (quarters + quarter rests @ 100 BPM)
python -m cello_note_detector data/multi_note_phrases/rhythm_quarters_with_rests.wav \
  -o outputs/rhythm --skip-mert
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--skip-mert` | pyin + rhythm only (no HF model download) |
| `--skip-rhythm` | notes only (no tempo / rests / duration values) |
| `--grid-beats 0.25` | snap grid in beats (default sixteenth) |
| `--min-rest-sec 0.08` | minimum silence gap emitted as a rest |
| `--no-edge-rests` | omit leading/trailing rests; keep inter-note gaps only |
| `--save-embeddings` | also write `[T, 768]` `.npy` next to the JSON |
| `--device cpu` | force CPU |
| `-o outputs` | output directory (default: `outputs/`) |
| `--no-onsets` | disable onset-based note splits |
| `--gap-tolerance 0.06` | bridge short unvoiced gaps inside one note |
| `--min-duration 0.04` | drop flicker shorter than this |

### Output

For each WAV, writes `outputs/<stem>_notes.json` with:

- `notes`: pyin events plus rhythm fields when enabled:
  - `onset_beats`, `duration_beats`
  - `quantized_onset_beats`, `quantized_duration_beats`
  - `duration_value` (e.g. `quarter`, `eighth`, `dotted_quarter`)
- `tempo`: `{bpm, beat_times, method, …}` beat grid estimate
- `rests`: silence gaps as quantized rest events (same rhythm fields as notes). **Inter-note gaps** are the primary rests; leading/trailing silence is included unless `--no-edge-rests`.
- `events`: chronological mix of `{type: "note"|"rest", …}` for a single timeline
- `num_notes` / `num_rests`
- `segmentation`: hop/frame settings, onset times, gap/median options
- `mert.summary`: embedding shape / mean / std (unless `--skip-mert`)

Example rhythm fields on a note / rest:

```json
{
  "onset_beats": 0.02,
  "duration_beats": 0.98,
  "quantized_onset_beats": 0.0,
  "quantized_duration_beats": 1.0,
  "duration_value": "quarter",
  "duration_value_beats": 1.0
}
```

Console prints note preview, rest preview, estimated BPM, and optional MERT shape.

## Multi-note tests

The sample-pack loops under `data/Recordings/` are mostly sustained single pitches. For **multi-note** and **rhythm/rest** checks, use the synthetic phrases in `data/multi_note_phrases/` (additive cello-range tones — license-clean; see that folder’s README).

```bash
# pyin + rhythm (fast)
python -m cello_note_detector data/multi_note_phrases/ -o outputs/multi_note --skip-mert

# one phrase
python -m cello_note_detector data/multi_note_phrases/scale_c_major_cello.wav -o outputs/multi_note --skip-mert
```

Expect `num_notes` > 1 (e.g. 8 for the C-major scale, 7 for Twinkle opening). Sibling `*_expected.json` files list the renderer’s ground-truth MIDI (and `expected_rests` when present).

Regenerate phrases after editing the generator:

```bash
python scripts/generate_multi_note_phrases.py
```

## Layout

```text
src/cello_note_detector/
  audio/          # load, pyin + onsets, MERT embed
  decode/         # note events + rhythm (tempo, quantize, rests)
  cli.py          # entrypoint
scripts/
  generate_multi_note_phrases.py
data/Recordings/          # sample-pack cello WAVs
data/multi_note_phrases/  # synthetic multi-note + rhythm test clips
outputs/                  # generated JSON / .npy (gitignored)
```

## Notes / limits

- Scope v1: **audio-only**, monophonic cello, **MERT-v1-95M** (not 330M).
- Transformers is pinned to **4.43.4** for MERT’s custom code.
- Segmentation splits on MIDI pitch changes, short gaps, and spectral onsets (so repeated same-pitch notes can separate). It is still a classical baseline — vibrato, slides, and noisy real recordings can merge/split notes incorrectly.
- Tempo/BPM is a **heuristic** (`beat_track` + note IOIs). Short clips, rubato, or single sustained drones can yield unstable or octave-ambiguous BPM; quantized `duration_value` labels follow that estimate.
- Rests come from **silence gaps**, not a trained rest detector. Gaps shorter than `--min-rest-sec` are ignored; very short articulation gaps may be missed or folded into note boundaries via `--gap-tolerance`.
- Out of scope for this milestone: polished MusicXML/sheet engraving, video, bowings/fingerings, MERT fine-tuning.
