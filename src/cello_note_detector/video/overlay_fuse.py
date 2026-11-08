"""Draw fused AV labels onto overlay frames (post-pass over the video)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from cello_note_detector.video.fusion import fusion_overlay_label


def write_fused_overlays(
    video_path: Path,
    fused: dict[str, Any],
    overlay_dir: Path,
    *,
    every: int = 10,
    max_writes: int | None = None,
) -> int:
    """Write annotated JPEGs for every Nth fused sample. Returns frames written."""
    overlay_dir.mkdir(parents=True, exist_ok=True)
    fused_samples = fused.get("fused_samples") or []
    if not fused_samples:
        return 0

    by_frame = {
        int(s["frame_idx"]): s
        for s in (fused.get("samples") or [])
        if s.get("frame_idx") is not None
    }
    # Subsample fused samples for overlay density.
    targets = [
        s
        for i, s in enumerate(fused_samples)
        if i % max(every, 1) == 0 and s.get("frame_idx") is not None
    ]
    if max_writes is not None:
        targets = targets[: max_writes]
    want = {int(s["frame_idx"]) for s in targets}
    fused_by_frame = {int(s["frame_idx"]): s for s in targets}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    written = 0
    frame_idx = 0
    while want:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx in want:
            fs = fused_by_frame[frame_idx]
            raw = by_frame.get(frame_idx, {})
            _draw(bgr, fs, raw)
            out_path = overlay_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), bgr)
            written += 1
            want.discard(frame_idx)
        frame_idx += 1

    cap.release()
    return written


def _draw(bgr, fused_sample: dict[str, Any], raw_sample: dict[str, Any]) -> None:
    h, w = bgr.shape[:2]

    def _dot(hand: dict[str, Any] | None, color: tuple[int, int, int]) -> None:
        if not hand:
            return
        wrist = hand.get("wrist") or {}
        if "x" in wrist and "y" in wrist:
            cv2.circle(bgr, (int(wrist["x"] * w), int(wrist["y"] * h)), 6, color, 2)
        for tip in (hand.get("tips") or {}).values():
            if tip and "x" in tip and "y" in tip:
                cv2.circle(bgr, (int(tip["x"] * w), int(tip["y"] * h)), 5, color, -1)

    _dot(raw_sample.get("left_hand"), (80, 200, 80))
    _dot(raw_sample.get("right_hand"), (80, 160, 255))

    line1 = f"t={fused_sample.get('t', 0):.2f}s"
    fingers = fused_sample.get("active_fingers") or []
    line1 += f"  curl={fingers}"
    line2 = fusion_overlay_label(fused_sample) or "(no pitch)"

    cv2.rectangle(bgr, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.putText(
        bgr, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA
    )
    cv2.putText(
        bgr, line2, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA
    )
