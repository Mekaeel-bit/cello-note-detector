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
| **`mspho_a_major_3oct_front.mp4`** | **Three Octave A major Scale Cello** (recommended AV-fusion demo; has audible cello) | Front-facing | https://www.youtube.com/watch?v=_XXFN5ZyKRk | _(none yet — fetch with `yt-dlp`)_ |
| `wmea_jr_scale_etude_front.mp4` | WMEA Jr cello **Scale Etude** (Dr. Ruth Boden / WSU) | Front / slight diagonal, landscape studio | https://www.youtube.com/watch?v=KZ4Id52zQw8 | https://archive.org/details/youtube-KZ4Id52zQw8 |
| `wmea_hs_chromatic_scale_front.mp4` | WMEA HS cello **Chromatic Scale** | Front / slight diagonal, landscape studio | https://www.youtube.com/watch?v=_jecuyxPO6s | https://archive.org/details/youtube-_jecuyxPO6s |
| `wmea_jr_scale_etude_intro_front.mp4` | Same Scale Etude **intro** talking + play | Front-facing vertical phone | https://www.youtube.com/watch?v=stMH0JKsqRQ | https://archive.org/details/youtube-stMH0JKsqRQ |
| `wmea_hs_chromatic_intro_front.mp4` | Chromatic scale **intro** talking + play | Front-facing vertical phone | https://www.youtube.com/watch?v=yNxSElVqKcM | https://archive.org/details/youtube-yNxSElVqKcM |

**Primary AV-fusion clip:** `mspho_a_major_3oct_front.mp4` — YouTube title *Three Octave A major Scale Cello* (channel **MsPhoOrchestra**; cellist Jeremiah Neely per channel branding). Bundled ~45 s **H.264 + stereo AAC-LC** (`-movflags +faststart`) so Windows players can hear it. Re-fetch locally:

```bash
yt-dlp -f "bv*+ba/b" --merge-output-format mp4 \
  -o data/video/raw/_XXFN5ZyKRk.%(ext)s \
  "https://www.youtube.com/watch?v=_XXFN5ZyKRk"
ffmpeg -y -i data/video/raw/_XXFN5ZyKRk.mp4 -t 45 \
  -vf "scale='min(1280,iw)':-2" \
  -c:v libx264 -pix_fmt yuv420p -profile:v high -movflags +faststart \
  -c:a aac -profile:a aac_low -ac 2 -ar 44100 -b:a 192k \
  data/video/clips/mspho_a_major_3oct_front.mp4
```

If the MP4 still plays silent in a given player, `--fuse` always writes `outputs/video/<stem>_audio.wav` (PCM) you can open directly; JSON records `audio_path` / `video_path` / `video_only_path`.

**Uploader / context (WMEA set):** Washington State University School of Music preparation tracks for WMEA auditions (“WMEA does not specifically endorse any preparation materials” per video descriptions). Some WMEA sandbox trims can look quiet / fail Windows fuse if `ffmpeg` is missing from PATH — prefer the MsPho A-major clip for pitch fusion.

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
