# cello-note-detector

Hackathon app for identifying cello notes from **monophonic audio** (video / fingerings later).

**Current milestone:** multi-note monophonic melody transcription (pyin + onsets) with **cleaner same-pitch / vibrato merge** → **tempo / key / beat grid + coarser quantized note durations and rests** → optional **song ID against a local public-domain catalog** (attach published MusicXML/MIDI on confident match) → optional **MusicXML / MIDI** export, plus optional **MERT-v1-95M** frame embeddings.

Tempo and key are inferred from **audio content** (spectral onsets + pitch histogram). They are **never** read from filenames. Optional `--bpm` / `--key` overrides are available when you already know them.

**Song ID policy (Shazam-like):** fingerprint audio → search the **local bundled** PD/CC MusicXML catalog → on miss, optionally **discover** remote PD seeds (`discovery_index.json`, IMSLP/Mutopia). Bach / Gabrielli / Dall'Abaco Caprices are **test** entries only — not a closed list (`scripts/add_catalog_piece.py`). Browse with `--list-catalog` / `--search-catalog`. **UX:** files under `data/catalog/pieces/` ship with the repo; they are **not** a live ID until `--identify` prints `MATCH` and copies an edition into `-o`. On miss / low confidence → **audio transcription** (never fails).

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
| `--identify` | Shazam-like ID (local catalog, then optional discover); implies `--score-source auto` |
| `--discover` / `--no-discover` | after local miss, try remote PD seeds (default on with identify) |
| `--discovery-index PATH` | remote seed JSON (default: `data/catalog/discovery_index.json`) |
| `--score-source auto\|transcribe\|catalog` | sheet source: catalog on confident match (`auto`), always transcription, or prefer catalog (still falls back) |
| `--catalog PATH` | bundled catalog JSON (default: `data/catalog/catalog.json`) |

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

## Song identification + published score (Shazam-like)

**Bundled ≠ identified.** `data/catalog/pieces/*` are PD/CC editions that **ship in the repo**. Opening those files (or seeing them in git) is not a live song ID. A real MATCH only happens when you run `--identify` and the CLI copies an edition into your `-o` folder.

Flow: audio fingerprint → **local catalog** → on miss, **`--discover`** (default) tries remote PD seeds in `data/catalog/discovery_index.json` → still miss → **transcription**.

Test entries (extensible — not a closed list): Bach Suite 1 Prélude, Gabrielli 1st Ricercar, **Dall'Abaco Capricci 1–2** (Joseph Marie **Clément** Dall'Abaco — not Clementi). Caprice 2 edition is **IMSLP PDF → Audiveris OMR** (Hamrefors engraved Caprice No.2 — not the complete Caprices book). On MATCH, primary `<stem>.musicxml` is the PD edition **clip excerpt** (whole movement when the take covers it); performance dump: `<stem>_transcribed.musicxml`.

**Open in MuseScore after identify:**

| Clip | Open this file |
|------|----------------|
| Bach long take | `outputs/bach_id/Bach-cello-suite-1.musicxml` |
| Gabrielli | `outputs/gabrielli_id/gabrielli.musicxml` |
| Dall'Abaco Caprices | `outputs/clement_id/clement-caprices.musicxml` **only if MATCH** |

(Do **not** open `*_transcribed.musicxml` for the PD edition.)

```bash
# Bach → verbatim Mutopia/Schirmer MusicXML
python -m cello_note_detector data/multi_note_phrases/Bach-cello-suite-1.wav \
  -o outputs/bach_id --skip-mert --quantize score --identify --musicxml --midi

# Gabrielli → catalog PD MusicXML
python -m cello_note_detector data/multi_note_phrases/gabrielli.wav \
  -o outputs/gabrielli_id --skip-mert --quantize score --identify --musicxml

# Clément Dall'Abaco Caprices test clip (local catalog + discover)
python -m cello_note_detector data/multi_note_phrases/clement-caprices.wav \
  -o outputs/clement_id --skip-mert --quantize score --identify --musicxml

# Local-only identify (no remote fetch attempts)
python -m cello_note_detector data/multi_note_phrases/clement-caprices.wav \
  -o outputs/clement_id --skip-mert --identify --no-discover --musicxml

# Fallback: unknown solo → transcription (no failure)
python -m cello_note_detector data/multi_note_phrases/scale_c_major_cello.wav \
  -o outputs/fallback --skip-mert --identify --musicxml --midi
```

Browse / look up the local PD index:

```bash
python -m cello_note_detector --list-catalog
python -m cello_note_detector --search-catalog dallabaco
```

Add any PD edition (MusicXML already available):

```bash
python scripts/add_catalog_piece.py \
  --id my_solo --musicxml /path/to/imslp.musicxml \
  --title "…" --composer "…" --license "Public Domain"
```

**PDF-only IMSLP** (engraved typeset preferred — reject manuscripts):

```bash
# 1) Fetch a single-movement engraved PDF (Hamrefors Caprice No.2 example)
python scripts/fetch_imslp_score.py \
  --work-url "https://imslp.org/wiki/11_Capricci_for_Cello_(Dall'Abaco,_Joseph)" \
  --prefer pdf --imslp-id 618567 --substring Caprice_No._2 \
  -o /tmp/caprice2.pdf

# 2) Audiveris OMR → MusicXML (requires Audiveris 5.x)
python scripts/pdf_to_musicxml.py /tmp/caprice2.pdf -o /tmp/caprice2_omr --fingerprint

# 3) Register in the extensible catalog (any cello PD piece)
python scripts/add_catalog_piece.py \
  --id dallabaco_capriccio2 \
  --musicxml /tmp/caprice2_omr/caprice2.musicxml \
  --title "Dall'Abaco Capriccio No.2 in G minor" \
  --composer "Joseph Marie Clément Dall'Abaco" \
  --edition "Hamrefors IMSLP #618567 via Audiveris OMR" \
  --key "G minor" --typical-duration-sec 176 --min-confidence 0.35
```

With `--discover`, PDF-only seeds in `discovery_index.json` (`omr_pdf: true`) fetch the engraved PDF and run Audiveris automatically when the piece is not already local.

| Outcome | Primary `<stem>.musicxml` | Also written | JSON |
|---------|---------------------------|--------------|------|
| Confident catalog/discovery MATCH | **PD edition clip** (full movement if take covers it) | `*_transcribed.musicxml` | `score_kind=verbatim_pd_edition` or `clip_excerpt_pd_edition` |
| No match / low confidence | Audio transcription | — | `fallback=audio_transcription` |
| Default (no `--identify`) | Transcription | — | no `identification` block |

## Bach Prelude accuracy sandbox

Two Prelude WAVs live under `data/multi_note_phrases/` (`Bach_Prelude_from_Cello_Suite_in_G.wav`, `Bach-cello-suite-1.wav`). Compare against the Mutopia/Schirmer public-domain monophonic reference in `data/reference/bwv1007_prelude/` (IMSLP-aligned pitch/rhythm; bowings non-authoritative). Prefer `--quantize score` for written equal-sixteenth intent. For demos that want the **published** score rather than the performance dump, use `--identify` (above). Never infer tempo/key from filenames.

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
  identify/       # local catalog song ID + PD score attach / fallback
  export/         # MusicXML + MIDI writers
  cli.py          # entrypoint
scripts/
  generate_multi_note_phrases.py
data/Recordings/          # sample-pack cello WAVs
data/multi_note_phrases/  # synthetic multi-note + rhythm test clips
data/catalog/             # local PD cello catalog index
data/reference/           # Mutopia/Schirmer (etc.) monophonic references
outputs/                  # generated JSON / MusicXML / MIDI / .npy (gitignored)
```

## Notes / limits

- Scope v1: **audio-only**, monophonic cello, **MERT-v1-95M** (not 330M).
- Transformers is pinned to **4.43.4** for MERT’s custom code.
- Segmentation splits on MIDI pitch changes, short gaps, and spectral onsets (so repeated same-pitch notes can separate). A post-pass merges short ±1-semitone vibrato flicker. Slides and noisy real recordings can still merge/split notes incorrectly.
- Tempo/BPM is a **heuristic** (spectral-onset IOIs + `beat_track`, with note-duration octave resolve). Short clips, rubato, or single sustained drones can still be octave-ambiguous; use `--bpm` when you know the pulse.
- Key is inferred from a duration-weighted pitch-class histogram (Krumhansl–Schmuckler). Sparse or heavily chromatic clips can pick the wrong mode/tonic; use `--key` to override. Filenames are never consulted.
- Rests come from **silence gaps**, not a trained rest detector. Gaps shorter than `--min-rest-sec` are ignored; very short articulation gaps may be missed or folded into note boundaries via `--gap-tolerance`.
- MusicXML/MIDI export is a **straight transcription dump** (4/4, inferred/override key signature, bass clef, sequential quantized events) unless song ID attaches a **catalog reference** score. It does not infer meter, engraver layout, or beaming polish. `--musicxml` / `--midi` require rhythm for transcription exports (they no-op with a warning under `--skip-rhythm`).
- Song ID is a **small local catalog** matcher (window accuracy + contiguous runs; ±1 octave). It is not a general music-recognition service. Only PD sources are shipped; commercial editions are out of scope. Unknown solos always fall back to transcription.
- **PDF OMR (Audiveris):** best on clean engraved single-movement PDFs. Manuscripts / handwritten scans are rejected by default. Dense tenor-clef cello writing and grace notes can still mis-read — spot-check MuseScore.
- **Caprices MATCH:** when pitch-contiguous gates fail but duration+key strongly pick a catalog movement (e.g. `clement-caprices.wav` → Caprice 2), identify **hard-promotes** to MATCH and attaches the OMR/IMSLP edition (clip excerpt / full movement) — not the garbage transcription MuseScore dump. Broader any-piece pitch ID is still longer.
- Out of scope for this milestone: video / fingerings / bowings, full Elgar concerto catalog, website `/transcribe` backend, MERT fine-tuning.
