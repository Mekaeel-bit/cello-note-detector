# Extensible PD score catalog (bundled + discoverable)

Local catalog for **song ID + verbatim published MusicXML attach**.

## Important UX (please read)

Files under `data/catalog/pieces/` are **bundled PD/CC editions that ship with
the repo**. Seeing Bach / Gabrielli / Dall'Abaco MusicXML on disk **before** you
run the CLI does **not** mean the audio was identified. A live ID looks like:

- Console: `identify: MATCH … score_kind=verbatim_pd_edition`
- Output: `outputs/<stem>.musicxml` is a **copy made during this run**
- JSON: `identification.matched=true`

On miss: transcription fallback (`*_notes.json` / transcribed MusicXML) — never
pretend a pre-existing catalog file is a Shazam hit.

## Shazam-like flow

1. Fingerprint audio (pitch sequence).
2. Search **local** `catalog.json` (extensible; not limited to 2 pieces).
3. On miss, `--discover` (default with `--identify`) consults
   `discovery_index.json` and tries live IMSLP/Mutopia/direct MusicXML URLs.
4. Still miss → audio transcription.

```bash
python -m cello_note_detector data/multi_note_phrases/clement-caprices.wav \
  -o outputs/clement_id --skip-mert --quantize score --identify --musicxml

# Local-only (no remote fetch attempts)
python -m cello_note_detector … --identify --no-discover
```

## Layout

```text
data/catalog/
  catalog.json                 # bundled index
  discovery_index.json         # remote PD seeds (live fetch on miss)
  pieces/<id>/
    edition.musicxml|.mxl      # VERBATIM PD edition (what MuseScore opens)
    fingerprint.json           # pitch sequence for matching
    meta.json
    SOURCE.txt                 # provenance
    source/                    # LilyPond / IMSLP sidecars
```

## Browse / look up

```bash
python -m cello_note_detector --list-catalog
python -m cello_note_detector --search-catalog dallabaco
python -m cello_note_detector --search-catalog Capriccio
```

## Add a piece

```bash
python scripts/add_catalog_piece.py \
  --id my_pd_solo \
  --musicxml /path/to/imslp_or_mutopia.musicxml \
  --title "Title" \
  --composer "Composer" \
  --license "Public Domain" \
  --imslp "https://imslp.org/wiki/..." \
  --typical-duration-sec 180
```

Prefer **MusicXML/MXL edition files**. Do not scrape commercial editions.

Current test entries: Bach BWV 1007 Prélude, Gabrielli 1st Ricercar,
Dall'Abaco Capricci Nos. 1–2 (Clément — **not** Clementi).
