# Extensible PD score catalog

Local catalog for **song ID + verbatim published MusicXML attach**.

Bach Suite 1 Prélude and Gabrielli Ricercari are the first test entries — the
catalog is **not** limited to them. Add any public-domain / CC0 cello MusicXML
(IMSLP, Mutopia, etc.).

## Layout

```text
data/catalog/
  catalog.json                 # index of pieces
  pieces/<id>/
    edition.musicxml|.mxl      # VERBATIM PD edition (what MuseScore opens)
    fingerprint.json           # pitch sequence + ornament stats for matching
    meta.json                  # title / license / IMSLP link
    source/                    # optional LilyPond / provenance sidecars
```

## Browse / look up

```bash
python -m cello_note_detector --list-catalog
python -m cello_note_detector --search-catalog bach
python -m cello_note_detector --search-catalog IMSLP
```

Audio identify searches the same index by pitch fingerprint; unknown audio
falls back to transcription.

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

## MATCH behavior

On confident identify:

- Primary `<stem>.musicxml` = **byte-for-byte copy** of `edition.musicxml`
  (slurs, trills, ties kept).
- `<stem>_transcribed.musicxml` = audio transcription (comparison only).

On miss / low confidence: transcription export only (never fails).

## Gabrielli CC0 MXL

When you can download IMSLP #923358 (Amaracle CC0), replace the placeholder:

```bash
python scripts/add_catalog_piece.py \
  --id gabrielli_ricercari \
  --musicxml ~/Downloads/IMSLP923358.mxl \
  --title "Gabrielli 7 Ricercari" \
  --composer "Domenico Gabrielli" \
  --license "CC0 1.0" \
  --imslp "https://imslp.org/wiki/Ricercari,_canone_e_sonate_per_violoncello_(Gabrielli,_Domenico)"
```
