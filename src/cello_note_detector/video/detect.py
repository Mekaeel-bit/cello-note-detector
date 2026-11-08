"""Hackathon fingering / bow detectors from front-facing cello video.

Uses MediaPipe Hands + Pose (Tasks API). Outputs are approximate:
- bow direction from right-wrist horizontal velocity (front camera)
- finger "press" proxies from left-hand fingertip extension toward the palm/wrist
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cello_note_detector.video.models import ensure_models

# MediaPipe hand landmark indices
_WRIST = 0
_THUMB_TIP = 4
_INDEX_MCP = 5
_INDEX_PIP = 6
_INDEX_TIP = 8
_MIDDLE_MCP = 9
_MIDDLE_PIP = 10
_MIDDLE_TIP = 12
_RING_MCP = 13
_RING_PIP = 14
_RING_TIP = 16
_PINKY_MCP = 17
_PINKY_PIP = 18
_PINKY_TIP = 20
# cello fingers 1–4 → index/middle/ring/pinky
_CELLO_FINGERS = {
    1: (_INDEX_MCP, _INDEX_PIP, _INDEX_TIP),
    2: (_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_TIP),
    3: (_RING_MCP, _RING_PIP, _RING_TIP),
    4: (_PINKY_MCP, _PINKY_PIP, _PINKY_TIP),
}

# Pose landmark indices (MediaPipe Pose)
_POSE_RIGHT_WRIST = 16
_POSE_LEFT_WRIST = 15
_POSE_RIGHT_ELBOW = 14
_POSE_LEFT_ELBOW = 13
_POSE_RIGHT_SHOULDER = 12
_POSE_LEFT_SHOULDER = 11


@dataclass
class FrameSample:
    t: float
    frame_idx: int
    left_hand: dict[str, Any] | None = None
    right_hand: dict[str, Any] | None = None
    bow: dict[str, Any] | None = None
    active_fingers: list[int] = field(default_factory=list)


def _hand_summary(landmarks, label: str, score: float) -> dict[str, Any]:
    tips = {}
    for name, idx in (
        ("thumb", _THUMB_TIP),
        ("index", _INDEX_TIP),
        ("middle", _MIDDLE_TIP),
        ("ring", _RING_TIP),
        ("pinky", _PINKY_TIP),
    ):
        lm = landmarks[idx]
        tips[name] = {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
    wrist = landmarks[_WRIST]
    return {
        "label": label,
        "score": round(float(score), 3),
        "wrist": {"x": round(wrist.x, 4), "y": round(wrist.y, 4)},
        "tips": tips,
    }


def _hand_scale(landmarks) -> float:
    wrist = landmarks[_WRIST]
    mcp = landmarks[_MIDDLE_MCP]
    return float(max(np.hypot(mcp.x - wrist.x, mcp.y - wrist.y), 1e-3))


def _active_cello_fingers(landmarks, *, curl_ratio_max: float = 0.72) -> list[int]:
    """Heuristic finger activity from curl (tip closer to MCP → more "down").

    Intentionally crude and not string/position aware. Returns cello finger
    numbers 1–4 (index→pinky). Open string (0) is inferred only when none
    look curled.
    """
    active: list[int] = []
    scale = _hand_scale(landmarks)
    for finger, (mcp_i, pip_i, tip_i) in _CELLO_FINGERS.items():
        mcp = landmarks[mcp_i]
        tip = landmarks[tip_i]
        tip_dist = float(np.hypot(tip.x - mcp.x, tip.y - mcp.y)) / scale
        # Curled / pressed fingers have a shorter tip–MCP span than extended ones.
        if tip_dist <= curl_ratio_max:
            active.append(finger)
    return active


def _assign_left_right(hand_res) -> tuple[Any, Any, Any, Any]:
    """Return (left_lms, left_summary, right_lms, right_summary).

    Prefer MediaPipe handedness; if only one hand or labels look swapped for a
    front-facing cellist, fall back to geometry (fingering hand is usually
    higher in the frame / smaller y).
    """
    pairs = []
    for lms, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
        label = handedness[0].category_name
        score = float(handedness[0].score)
        pairs.append((lms, label, score, _hand_summary(lms, label, score)))

    left_lms = right_lms = None
    left_sum = right_sum = None
    for lms, label, score, summary in pairs:
        if label.lower().startswith("left"):
            left_lms, left_sum = lms, summary
        else:
            right_lms, right_sum = lms, summary

    if len(pairs) == 2 and (left_lms is None or right_lms is None):
        # Ambiguous labels — use vertical position.
        pairs_sorted = sorted(pairs, key=lambda p: p[0][_WRIST].y)
        left_lms, _, _, left_sum = pairs_sorted[0]
        right_lms, _, _, right_sum = pairs_sorted[1]
        left_sum = dict(left_sum)
        right_sum = dict(right_sum)
        left_sum["label"] = "Left(geometry)"
        right_sum["label"] = "Right(geometry)"
    elif len(pairs) == 1:
        lms, label, score, summary = pairs[0]
        # Single hand: if high in frame treat as left (fingering), else bow.
        if lms[_WRIST].y < 0.55:
            left_lms, left_sum = lms, summary
        else:
            right_lms, right_sum = lms, summary
    elif len(pairs) == 2 and left_lms is not None and right_lms is not None:
        # If MediaPipe left wrist is clearly below right, swap (common flip).
        if left_lms[_WRIST].y > right_lms[_WRIST].y + 0.08:
            left_lms, right_lms = right_lms, left_lms
            left_sum, right_sum = right_sum, left_sum
            left_sum = dict(left_sum)
            right_sum = dict(right_sum)
            left_sum["label"] = "Left(swapped)"
            right_sum["label"] = "Right(swapped)"

    return left_lms, left_sum, right_lms, right_sum


def _bow_from_right_wrist(
    xs: list[float],
    ys: list[float],
    ts: list[float],
    *,
    vel_eps: float = 0.08,
) -> dict[str, Any] | None:
    if len(xs) < 3:
        return None
    # Smooth over last few samples to reduce flicker.
    dx = xs[-1] - xs[-3]
    dy = ys[-1] - ys[-3]
    dt = max(ts[-1] - ts[-3], 1e-3)
    vx, vy = dx / dt, dy / dt
    direction = "hold"
    if abs(vx) >= vel_eps:
        # Image-space: negative vx (leftward) ≈ toward frog side of frame for
        # typical front shots — labeled down/up heuristically.
        direction = "down" if vx < 0 else "up"
    return {
        "direction": direction,
        "vx": round(vx, 4),
        "vy": round(vy, 4),
        "wrist_x": round(xs[-1], 4),
        "wrist_y": round(ys[-1], 4),
        "position": "frog" if xs[-1] < 0.35 else ("tip" if xs[-1] > 0.55 else "middle"),
    }


def analyze_video(
    video_path: Path,
    *,
    model_dir: Path | None = None,
    sample_every: int = 2,
    max_frames: int | None = None,
    overlay_dir: Path | None = None,
    overlay_every: int = 10,
) -> dict[str, Any]:
    """Run hand+pose tracking; return JSON-serializable timeline."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    models = ensure_models(model_dir)
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(models["hand_landmarker.task"])),
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=vision.RunningMode.VIDEO,
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(models["pose_landmarker_lite.task"])
        ),
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=vision.RunningMode.VIDEO,
    )
    hands = vision.HandLandmarker.create_from_options(hand_opts)
    pose = vision.PoseLandmarker.create_from_options(pose_opts)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    samples: list[FrameSample] = []
    events: list[dict[str, Any]] = []
    prev_fingers: set[int] = set()
    prev_bow_dir: str | None = None
    right_xs: list[float] = []
    right_ys: list[float] = []
    right_ts: list[float] = []

    frame_idx = 0
    processed = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and processed >= max_frames:
            break
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        t = frame_idx / fps
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(t * 1000)

        hand_res = hands.detect_for_video(mp_image, ts_ms)
        pose_res = pose.detect_for_video(mp_image, ts_ms)

        left_lms, left_sum, right_lms, right_sum = _assign_left_right(hand_res)
        active: list[int] = []
        if left_lms is not None:
            active = _active_cello_fingers(left_lms)

        # Prefer pose right wrist for bow if hands miss the bow grip
        wrist_x = wrist_y = None
        if right_lms is not None:
            wrist_x = float(right_lms[_WRIST].x)
            wrist_y = float(right_lms[_WRIST].y)
        elif pose_res.pose_landmarks:
            pl = pose_res.pose_landmarks[0]
            rw = pl[_POSE_RIGHT_WRIST]
            if rw.visibility > 0.4:
                wrist_x, wrist_y = float(rw.x), float(rw.y)

        bow = None
        if wrist_x is not None and wrist_y is not None:
            right_xs.append(wrist_x)
            right_ys.append(wrist_y)
            right_ts.append(t)
            if len(right_xs) > 10:
                right_xs.pop(0)
                right_ys.pop(0)
                right_ts.pop(0)
            bow = _bow_from_right_wrist(right_xs, right_ys, right_ts)

        sample = FrameSample(
            t=round(t, 3),
            frame_idx=frame_idx,
            left_hand=left_sum,
            right_hand=right_sum,
            bow=bow,
            active_fingers=active,
        )
        samples.append(sample)

        # Finger change events
        cur = set(active)
        for f in sorted(cur - prev_fingers):
            events.append(
                {
                    "t": round(t, 3),
                    "type": "finger_on",
                    "finger": f,
                    "confidence": 0.4,
                    "note": "heuristic curl proxy; not string-aware",
                }
            )
        for f in sorted(prev_fingers - cur):
            events.append(
                {
                    "t": round(t, 3),
                    "type": "finger_off",
                    "finger": f,
                    "confidence": 0.35,
                }
            )
        if not cur and prev_fingers:
            events.append(
                {
                    "t": round(t, 3),
                    "type": "finger_open",
                    "finger": 0,
                    "confidence": 0.3,
                    "note": "no curled fingers → possible open string",
                }
            )
        prev_fingers = cur

        # Bow direction change with short hold to limit flicker
        if bow is not None:
            d = bow["direction"]
            if d in ("up", "down") and d != prev_bow_dir:
                # require stable direction across recent bow samples
                recent = [
                    s.bow["direction"]
                    for s in samples[-3:]
                    if s.bow and s.bow["direction"] in ("up", "down")
                ]
                if len(recent) >= 2 and all(x == d for x in recent[-2:]):
                    events.append(
                        {
                            "t": round(t, 3),
                            "type": "bow",
                            "direction": d,
                            "position": bow["position"],
                            "confidence": 0.5,
                            "note": "from right-wrist image-x velocity (front camera)",
                        }
                    )
                    prev_bow_dir = d

        if overlay_dir is not None and (processed % overlay_every == 0):
            _draw_overlay(bgr, sample, left_lms, right_lms)
            out_path = overlay_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), bgr)

        frame_idx += 1
        processed += 1

    cap.release()
    hands.close()
    pose.close()

    duration = (n_frames / fps) if n_frames else (frame_idx / fps)
    return {
        "schema": "cello_video_timeline_v0",
        "source": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3),
        "sample_every": sample_every,
        "frames_sampled": len(samples),
        "limits": [
            "Hackathon prototype — not concert accuracy.",
            "Finger indices are activity proxies, not scored fingerings on strings.",
            "Bow up/down is image-space wrist motion, seating/camera dependent.",
            "No string identity / position / shift detection yet.",
        ],
        "events": events,
        "samples": [
            {
                "t": s.t,
                "frame_idx": s.frame_idx,
                "active_fingers": s.active_fingers,
                "bow": s.bow,
                "left_hand": s.left_hand,
                "right_hand": s.right_hand,
            }
            for s in samples
        ],
    }


def _draw_overlay(bgr, sample: FrameSample, left_lms, right_lms) -> None:
    h, w = bgr.shape[:2]

    def _pt(lm):
        return int(lm.x * w), int(lm.y * h)

    for lms, color in ((left_lms, (80, 200, 80)), (right_lms, (80, 160, 255))):
        if lms is None:
            continue
        for tip_idx in (_THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP, _RING_TIP, _PINKY_TIP):
            cv2.circle(bgr, _pt(lms[tip_idx]), 5, color, -1)
        cv2.circle(bgr, _pt(lms[_WRIST]), 6, color, 2)

    label = f"t={sample.t:.2f}s  fingers={sample.active_fingers}"
    if sample.bow:
        label += f"  bow={sample.bow['direction']}/{sample.bow['position']}"
    cv2.rectangle(bgr, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


def write_timeline(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
