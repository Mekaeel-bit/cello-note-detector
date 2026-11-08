# Local score catalog (song ID)

Hackathon v1 catalog for **song identification + published score attach**.

Only **public-domain** editions belong here (Mutopia / IMSLP-aligned sources
already under `data/reference/`). Do **not** scrape copyrighted commercial
editions. Online fetch is optional and not required for v1 — local JSON is enough.

| File | Role |
|------|------|
| `catalog.json` | Index of known cello solos (`id`, title, edition, paths, thresholds) |

## Adding a piece

1. Drop a monophonic reference under `data/reference/<piece_id>/` with:
   - `midi_sequence`: list of MIDI ints (used for song-ID matching)
   - optional playable `score_json` with `events` for MusicXML/MIDI attach
     (prefer a clean edition file if the match reference is eval-only)
   - provenance fields (`title`, `source_edition`, `imslp_work`, `key`, …)
2. Append an entry to `catalog.json` pointing at `reference_json` and optional `score_json`.
3. Keep `min_confidence` conservative so unknown solos fall back to audio transcription.

On match, the primary `<stem>.musicxml` is always the catalog edition; the
performance dump is written beside it as `<stem>_transcribed.musicxml`.

## Fallback policy

If confidence is below the entry/catalog threshold, or the catalog is empty,
the CLI **always** keeps the audio transcription export. Missing public scores
must never fail a run.
