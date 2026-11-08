# Video sources (fingerings / bowings sandbox)

Hackathon demo footage for **front-facing** cello clips where the camera can see the bow and left-hand fingers. Prefer scales / audition etudes.

## License / usage note

These clips originate on **YouTube** (standard YouTube Terms of Service; creators retain rights — typically **All Rights Reserved**, not Creative Commons). They are used here **only for non-commercial R&D / hackathon prototyping**.

- Do **not** redistribute the binaries outside this repo’s sandbox use.
- Prefer re-downloading via the documented URLs rather than treating the repo copies as a distribution channel.
- If a creator objects, remove the clip and switch to self-recorded / clearly licensed footage.

Direct `yt-dlp` from this cloud VM was bot-gated (“Sign in to confirm you’re not a bot”). Clips were obtained via **Internet Archive YouTube item mirrors** that point back to the same watch URLs.

## Clips in `data/video/clips/`

Trimmed H.264 demos (title cards removed; ~18–25 s each) for a runnable sandbox:

| Local file | Content | Angle | YouTube | Archive.org mirror |
|------------|---------|-------|---------|--------------------|
| `wmea_jr_scale_etude_front.mp4` | WMEA Jr cello **Scale Etude** (Dr. Ruth Boden / WSU) | Front / slight diagonal, landscape studio | https://www.youtube.com/watch?v=KZ4Id52zQw8 | https://archive.org/details/youtube-KZ4Id52zQw8 |
| `wmea_hs_chromatic_scale_front.mp4` | WMEA HS cello **Chromatic Scale** | Front / slight diagonal, landscape studio | https://www.youtube.com/watch?v=_jecuyxPO6s | https://archive.org/details/youtube-_jecuyxPO6s |
| `wmea_jr_scale_etude_intro_front.mp4` | Same Scale Etude **intro** talking + play | Front-facing vertical phone | https://www.youtube.com/watch?v=stMH0JKsqRQ | https://archive.org/details/youtube-stMH0JKsqRQ |
| `wmea_hs_chromatic_intro_front.mp4` | Chromatic scale **intro** talking + play | Front-facing vertical phone | https://www.youtube.com/watch?v=yNxSElVqKcM | https://archive.org/details/youtube-yNxSElVqKcM |

**Uploader / context:** Washington State University School of Music preparation tracks for WMEA auditions (“WMEA does not specifically endorse any preparation materials” per video descriptions).

## Optional candidates (not bundled; good next downloads)

Front-facing instructional scales (YouTube ToS; download locally with `yt-dlp` if your network allows):

| Title | URL | Notes |
|-------|-----|-------|
| Scales Workshop – Cello Grade 1 – C Major 2 octaves | https://www.youtube.com/watch?v=Yf8uA7FkhJM | Clear fingering callouts + bows |
| Cello Scales with the “Universal Fingering” | https://www.youtube.com/watch?v=TAeX274-l-E | Scale fingering demo |
| C Major Scale Stuff – Beginner Cello Tutorial | https://www.youtube.com/watch?v=qP0JJkRBEpI | Front-ish beginner C major |

Re-fetch the bundled set (IA mirrors):

```bash
python scripts/download_video_demos.py
```

## Models

MediaPipe `.task` bundles are **not** committed (large). First run of the video CLI downloads them into `data/video/models/` (gitignored), or:

```bash
python scripts/download_video_demos.py --models-only
```
