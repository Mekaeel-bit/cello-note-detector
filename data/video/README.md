# Video sandbox (fingerings / bowings)

Prototype milestone: estimate **left-hand finger activity** and **bow direction/position** over time from a **static front-facing** cello video, optionally **fused with pyin pitch** for string-aware labels.

Hackathon-honest: useful timeline + overlay for demos, **not** concert-grade fingering numbers or bowing marks.

See [`SOURCES.md`](SOURCES.md) for YouTube URLs, Archive.org mirrors, and license notes.

## Quick start

```bash
# optional video extras (OpenCV + MediaPipe)
pip install -r requirements-video.txt

# analyze a bundled clip → JSON timeline + overlay frames
python -m cello_note_detector.video data/video/clips/wmea_jr_scale_etude_front.mp4 \
  -o outputs/video --overlay --max-frames 180

# AV fusion (recommended): pyin pitch → string/position ∩ hand landmarks
python -m cello_note_detector.video data/video/clips/wmea_jr_scale_etude_front.mp4 \
  -o outputs/video --fuse --overlay --max-frames 180

# align roughly to audio only (extracts WAV via ffmpeg, runs pyin; no string fusion)
python -m cello_note_detector.video data/video/clips/wmea_hs_chromatic_scale_front.mp4 \
  -o outputs/video --overlay --align-audio
```

Outputs under `-o`:

- `<stem>_video_timeline.json` — frame samples + fingering/bow **events**
- `<stem>_fused_timeline.json` — with `--fuse`: string-aware fingering + bow string
- `<stem>_fused_sample.json` — short preview of fused notes/events
- `<stem>_overlay/` or `<stem>_fused_overlay/` — optional annotated JPEG frames

Example overlay stills (checked in): `demo_overlays/`.

### Fusion model (v0)

1. Extract mono WAV (ffmpeg) → existing `transcribe_notes` pyin pipeline.
2. Map each sounding MIDI to open-string/position hypotheses on **C2 / G2 / D3 / A3**.
3. Re-rank with left-hand neck-height + finger-curl proxies from MediaPipe.
4. Bow **direction** stays right-wrist image-x velocity; bow **string** = pitch hypothesis.
