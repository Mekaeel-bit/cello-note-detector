# BWV 1007 Prélude — pitch/rhythm reference

Public-domain **pitch + rhythm** ground truth for the Bach Cello Suite No. 1
Prelude accuracy sandbox.

| File | Role |
|------|------|
| `prelude_monophonic_reference.json` | Monophonic MIDI sequence + beat durations (primary eval target) |
| `prelude_midi_sequence.json` | Compact pitch-only sequence |
| `mutopia517_prelude.mid` | Mutopia compiler MIDI (may contain held-string overlaps — prefer the JSON) |
| `mutopia517_prelude.ly` | Mutopia LilyPond source (Schirmer 1916) |

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
