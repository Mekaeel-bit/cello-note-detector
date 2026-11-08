# Video sandbox (fingerings / bowings)

Prototype milestone: estimate **left-hand finger activity** and **bow direction/position** over time from a **static front-facing** cello video.

Hackathon-honest: useful timeline + overlay for demos, **not** concert-grade fingering numbers or bowing marks.

See [`SOURCES.md`](SOURCES.md) for YouTube URLs, Archive.org mirrors, and license notes.

## Quick start

```bash
# optional video extras (OpenCV + MediaPipe)
pip install -r requirements-video.txt

# analyze a bundled clip → JSON timeline + overlay frames
python -m cello_note_detector.video data/video/clips/wmea_jr_scale_etude_front.mp4 \
  -o outputs/video --overlay --max-frames 180

# align roughly to audio (extracts WAV via ffmpeg, runs pyin if available)
python -m cello_note_detector.video data/video/clips/wmea_hs_chromatic_scale_front.mp4 \
  -o outputs/video --overlay --align-audio
```

Outputs under `-o`:

- `<stem>_video_timeline.json` — frame samples + fingering/bow **events**
- `<stem>_overlay/` — optional annotated JPEG frames (every Nth frame)

Example overlay stills (checked in): `demo_overlays/`.
