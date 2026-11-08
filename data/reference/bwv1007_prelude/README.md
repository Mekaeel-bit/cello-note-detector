# BWV 1007 Prélude — pitch/rhythm reference

Public-domain **pitch + rhythm** ground truth for the Bach Cello Suite No. 1
Prelude accuracy sandbox.

| File | Role |
|------|------|
| `prelude_catalog_score.json` | **Playable** monophonic edition for MusicXML/MIDI attach (re-parsed from LilyPond) |
| `prelude_monophonic_reference.json` | Eval target (hand-checked prefix; later notes are best-effort — do not use for MuseScore) |
| `prelude_midi_sequence.json` | Compact pitch-only sequence |
| `mutopia517_prelude.mid` | Mutopia compiler MIDI (may contain held-string overlaps — prefer the catalog score JSON) |
| `mutopia517_prelude.ly` | Mutopia LilyPond source (Schirmer 1916) |

Regenerate the playable catalog score after editing the LilyPond:

```bash
python scripts/build_bwv1007_catalog_score.py
```

## Provenance

- **Edition:** Mutopia-2018/01/19-517 after **Schirmer, 1916** (Public Domain).
- **IMSLP work page:** [Cello Suite No.1 in G major, BWV 1007](https://imslp.org/wiki/Cello_Suite_No.1_in_G_major,_BWV_1007_(Bach,_Johann_Sebastian))
- Bach’s autograph is lost; editions differ (especially bowings). This reference
  is for **written pitch/rhythm**. Treat bowings/slurs as non-authoritative.
- LilyPond is written at `c'` and transposed `c → c,` for cello sounding pitch.
  Polyphonic `<< >>` inserts near the end were dropped from the monophonic JSON.

## Test WAVs

Teddy’s Prelude recordings live under `data/multi_note_phrases/`:

- `Bach_Prelude_from_Cello_Suite_in_G.wav` — short excerpt (~19 s)
- `Bach-cello-suite-1.wav` — longer take (~159 s)

Never infer tempo/key from those filenames.
