# cello-note-detector

Hackathon app for identifying cello notes from **monophonic audio**, plus a **video fingerings/bowings sandbox**.

**Current audio milestone:** multi-note monophonic melody transcription (pyin + onsets) with **cleaner same-pitch / vibrato merge** → **tempo / key / beat grid + coarser quantized note durations and rests** → optional **MusicXML / MIDI** export, plus optional **MERT-v1-95M** frame embeddings.

**Video sandbox:** front-facing cello clips → MediaPipe hand/pose timeline, optionally **fused with pyin pitch** for string-aware fingering + bow-string labels (hackathon-honest, not concert accuracy). See [`data/video/README.md`](data/video/README.md).

Tempo and key are inferred from **audio content** (spectral onsets + pitch histogram). They are **never** read from filenames. Optional `--bpm` / `--key` overrides are available when you already know them.

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

### Rhythm analysis (tempo + key + quantization + rests)

Rhythm runs **by default** after note detection. Defaults favour readable scores (quarters / eighths) over fussy 16th/32nd grids:

```bash
# Multi-note phrases (recommended for rhythm checks)
python -m cello_note_detector data/multi_note_phrases/ -o outputs/rhythm --skip-mert

# Dedicated rest clip (quarters + quarter rests @ 100 BPM)
python -m cello_note_detector data/multi_note_phrases/rhythm_quarters_with_rests.wav \
  -o outputs/rhythm --skip-mert

# Same clip → JSON + MusicXML + MIDI
python -m cello_note_detector data/multi_note_phrases/rhythm_quarters_with_rests.wav \
  -o outputs/sheet --skip-mert --musicxml --midi

# Optional overrides (still never read from the filename)
python -m cello_note_detector data/multi_note_phrases/120_Gm_Cello_33_189_SP.wav \
  -o outputs/sheet --skip-mert --musicxml --bpm 120 --key Gm --quantize coarse
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--skip-mert` | pyin + rhythm only (no HF model download) |
| `--skip-rhythm` | notes only (no tempo / rests / duration values) |
| `--bpm 120` | override estimated tempo (BPM) |
| `--key Gm` | override inferred key (`Gm`, `Bb`, `D major`, `A minor`, …) |
| `--quantize medium` | coarseness preset: `coarse` / `medium` (default) / `fine` / `score` |
| `--musicxml` | also write bass-clef `.musicxml` next to the JSON |
| `--midi` | also write Type-1 `.mid` next to the JSON |
| `--grid-beats 0.5` | override snap grid in beats (preset default: eighth) |
| `--min-rest-sec 0.16` | minimum silence gap emitted as a rest |
| `--no-edge-rests` | omit leading/trailing rests; keep inter-note gaps only |
| `--save-embeddings` | also write `[T, 768]` `.npy` next to the JSON |
| `--device cpu` | force CPU |
| `-o outputs` | output directory (default: `outputs/`) |
| `--no-onsets` | disable onset-based note splits |
| `--gap-tolerance 0.10` | bridge short unvoiced gaps inside one note |
| `--min-duration 0.08` | drop flicker shorter than this |
| `--median-filter-size 11` | MIDI median window to reduce vibrato flicker |

### Quantize presets

| Preset | Grid | Duration catalog | Typical use |
|--------|------|------------------|-------------|
| `coarse` | eighth | whole → eighth (no 16ths/32nds) | readable MuseScore drafts |
| `medium` (default) | eighth | + sixteenths | balanced |
| `fine` | sixteenth | + dotted 16ths / 32nds | old dense behaviour |
| `score` | sixteenth | sixteenths+ (equal-stream snap) | written-intent / moto-perpetuo reconstruction |

### Output

For each WAV, writes `outputs/<stem>_notes.json` with:

- `notes`: pyin events plus rhythm fields when enabled:
  - `onset_beats`, `duration_beats`
  - `quantized_onset_beats`, `quantized_duration_beats`
  - `duration_value` (e.g. `quarter`, `eighth`, `dotted_quarter`)
- `tempo`: `{bpm, beat_times, method, spectral_onset_bpm, …}` beat grid estimate
- `key`: `{name, tonic, mode, fifths, method, confidence}` inferred (or `--key`) signature
- `rests`: silence gaps as quantized rest events (same rhythm fields as notes). **Inter-note gaps** are the primary rests; leading/trailing silence is included unless `--no-edge-rests`.
- `events`: chronological mix of `{type: "note"|"rest", …}` for a single timeline
- `num_notes` / `num_rests`
- `segmentation`: hop/frame settings, onset times, gap/median / flicker-merge options
- `mert.summary`: embedding shape / mean / std (unless `--skip-mert`)

Optional sheet artifacts (same stem, next to the JSON):

| Flag | File | Contents |
|------|------|----------|
| `--musicxml` | `<stem>.musicxml` | Bass-clef partwise MusicXML 3.1 (notes + rests + tempo + key) |
| `--midi` | `<stem>.mid` | Type-1 MIDI from the quantized timeline (rests = silence) |

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

Console prints note preview, rest preview, estimated BPM, key, export paths, and optional MERT shape.

### Opening MusicXML

1. Run with `--musicxml` (rhythm must stay enabled — do not pass `--skip-rhythm`).
2. Open the `.musicxml` file in **[MuseScore](https://musescore.org/)** (File → Open), or Dorico / Finale / Sibelius / flat.io.
3. You should see a **bass-clef** cello staff in 4/4 with the estimated metronome mark and inferred key signature. Rests appear as rest glyphs, not missing notes.

MIDI (`.mid`) opens in the same apps or any DAW for a quick playback sanity check.

## Video fingerings / bowings sandbox

Bundled front-facing WMEA scale/etude clips (YouTube originals; IA mirrors — see [`data/video/SOURCES.md`](data/video/SOURCES.md)):

```bash
pip install -r requirements-video.txt
python -m cello_note_detector.video data/video/clips/wmea_jr_scale_etude_front.mp4 \
  -o outputs/video --overlay --max-frames 180
# AV fusion: pyin pitch ∩ landmarks → string + fingering + bow string
python -m cello_note_detector.video data/video/clips/wmea_jr_scale_etude_front.mp4 \
  -o outputs/video --fuse --overlay --max-frames 180
# optional rough AV sync without fusion
python -m cello_note_detector.video data/video/clips/wmea_hs_chromatic_scale_front.mp4 \
  -o outputs/video --overlay --align-audio
```

Writes `<stem>_video_timeline.json` (events + per-frame samples), with `--fuse` also `<stem>_fused_timeline.json` + sample preview, and optional overlay JPEGs. Models download on first run into `data/video/models/`.

## Bach Prelude accuracy sandbox

Two Prelude WAVs live under `data/multi_note_phrases/` (`Bach_Prelude_from_Cello_Suite_in_G.wav`, `Bach-cello-suite-1.wav`). Compare against the Mutopia/Schirmer public-domain monophonic reference in `data/reference/bwv1007_prelude/` (IMSLP-aligned pitch/rhythm; bowings non-authoritative). Prefer `--quantize score` for written equal-sixteenth intent. Never infer tempo/key from filenames.

```bash
python -m cello_note_detector data/multi_note_phrases/Bach_Prelude_from_Cello_Suite_in_G.wav \
  -o outputs/bach --skip-mert --quantize score --musicxml --midi
python scripts/eval_pitch_sequence.py outputs/bach/Bach_Prelude_from_Cello_Suite_in_G_notes.json
```

## Multi-note tests

The sample-pack loops under `data/Recordings/` are mostly sustained single pitches. For **multi-note** and **rhythm/rest** checks, use the synthetic phrases in `data/multi_note_phrases/` (additive cello-range tones — license-clean; see that folder’s README). Real multi-note sample-pack clips in that folder are also good regression checks — compare by ear/structure, and do **not** trust tempo/key tokens in the filename.

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
  decode/         # note events + rhythm (tempo, key, quantize, rests)
  export/         # MusicXML + MIDI writers
  video/          # MediaPipe fingering/bow sandbox CLI
  cli.py          # audio entrypoint
scripts/
  generate_multi_note_phrases.py
  download_video_demos.py
data/Recordings/          # sample-pack cello WAVs
data/multi_note_phrases/  # synthetic multi-note + rhythm test clips
data/video/               # front-facing demo clips + SOURCES.md
outputs/                  # generated JSON / MusicXML / MIDI / .npy (gitignored)
```

## Notes / limits

- Scope v1 audio: monophonic cello, **MERT-v1-95M** (not 330M).
- Transformers is pinned to **4.43.4** for MERT’s custom code.
- Segmentation splits on MIDI pitch changes, short gaps, and spectral onsets (so repeated same-pitch notes can separate). A post-pass merges short ±1-semitone vibrato flicker. Slides and noisy real recordings can still merge/split notes incorrectly.
- Tempo/BPM is a **heuristic** (spectral-onset IOIs + `beat_track`, with note-duration octave resolve). Short clips, rubato, or single sustained drones can still be octave-ambiguous; use `--bpm` when you know the pulse.
- Key is inferred from a duration-weighted pitch-class histogram (Krumhansl–Schmuckler). Sparse or heavily chromatic clips can pick the wrong mode/tonic; use `--key` to override. Filenames are never consulted.
- Rests come from **silence gaps**, not a trained rest detector. Gaps shorter than `--min-rest-sec` are ignored; very short articulation gaps may be missed or folded into note boundaries via `--gap-tolerance`.
- MusicXML/MIDI export is a **straight transcription dump** (4/4, inferred/override key signature, bass clef, sequential quantized events). It does not infer meter, engraver layout, or beaming polish. `--musicxml` / `--midi` require rhythm (they no-op with a warning under `--skip-rhythm`).
- Video sandbox: MediaPipe finger/bow proxies only; no string ID, shift detection, or engraved bowing marks yet. Demo clips are YouTube-sourced (ToS) via IA mirrors — see `data/video/SOURCES.md`.
- Out of scope here: song-ID / IMSLP catalog work, website `/transcribe` backend, MERT fine-tuning.
