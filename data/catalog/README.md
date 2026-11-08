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
   - `midi_sequence`: list of MIDI ints
   - `events`: `{midi, onset_beats, duration_beats, duration_value}` for score export
   - provenance fields (`title`, `source_edition`, `imslp_work`, `key`, …)
2. Append an entry to `catalog.json` pointing at that JSON.
3. Keep `min_confidence` conservative so unknown solos fall back to audio transcription.

## Fallback policy

If confidence is below the entry/catalog threshold, or the catalog is empty,
the CLI **always** keeps the audio transcription export. Missing public scores
must never fail a run.
