"""Fuse pyin pitch with video landmarks → string-aware fingering + bow string."""

from __future__ import annotations

from typing import Any

from cello_note_detector.video.strings import (
    STRING_ORDER,
    StringHypothesis,
    pitch_string_hypotheses,
    primary_hypothesis,
)


def _note_at_time(notes: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    """Return the sounding note covering time ``t`` (inclusive onset, exclusive offset)."""
    best: dict[str, Any] | None = None
    for n in notes:
        onset = float(n.get("onset", 0.0))
        offset = float(n.get("offset", onset))
        if onset <= t < offset:
            if best is None or onset >= float(best.get("onset", -1)):
                best = n
    return best


def _hand_neck_proxy(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Crude fingerboard-height proxy from left-hand landmarks (front camera).

    Smaller image-y ≈ toward the scroll / nut; larger y ≈ toward the bridge.
    Returns normalized height in [0, 1] where 0 ≈ nut-side, 1 ≈ bridge-side
    within the observed left-hand tip/wrist cluster for that frame.
    """
    left = sample.get("left_hand")
    if not left:
        return None
    wrist = left.get("wrist") or {}
    tips = left.get("tips") or {}
    ys = []
    if "y" in wrist:
        ys.append(float(wrist["y"]))
    for name in ("index", "middle", "ring", "pinky"):
        tip = tips.get(name)
        if tip and "y" in tip:
            ys.append(float(tip["y"]))
    if not ys:
        return None
    # Front-facing cellist: left hand typically occupies ~y 0.15–0.55.
    # Map into [0,1] neck proxy (0=nut/high in frame, 1=bridge/lower).
    y_mean = sum(ys) / len(ys)
    nut, bridge = 0.12, 0.55
    height = (y_mean - nut) / max(bridge - nut, 1e-3)
    height = max(0.0, min(1.0, height))
    return {
        "y_mean": round(y_mean, 4),
        "neck_height": round(height, 4),
        "band": (
            "open_or_nut"
            if height < 0.25
            else ("low" if height < 0.45 else ("mid" if height < 0.7 else "high"))
        ),
    }


def _vision_score(hyp: StringHypothesis, neck: dict[str, Any] | None, active: list[int]) -> float:
    """Lower is better. Blend pitch-rank with hand height + curl agreement."""
    score = float(hyp.rank_score)
    if neck is not None:
        band = neck["band"]
        # Reward matching position band vs hand height.
        if hyp.position_band == "open" and band == "open_or_nut":
            score -= 1.5
        elif hyp.position_band == band:
            score -= 0.8
        elif hyp.position_band == "low" and band == "open_or_nut":
            score -= 0.3
        elif hyp.position_band == "high" and band in ("mid", "high"):
            score -= 0.4
        else:
            score += 0.6

    if hyp.suggested_finger == 0:
        if not active:
            score -= 1.0
        else:
            score += 0.8
    else:
        if hyp.suggested_finger in active:
            score -= 1.2
        elif active:
            # Any curl present but wrong finger — mild penalty.
            score += 0.3
        else:
            # Expected a stop but saw open hand.
            score += 0.9
    return score


def _choose_hypothesis(
    midi: int,
    sample: dict[str, Any] | None,
) -> tuple[StringHypothesis | None, list[dict[str, Any]], dict[str, Any] | None]:
    hyps = pitch_string_hypotheses(midi)
    if not hyps:
        return None, [], None
    neck = _hand_neck_proxy(sample) if sample else None
    active = list((sample or {}).get("active_fingers") or [])
    scored = sorted(
        (( _vision_score(h, neck, active), h) for h in hyps),
        key=lambda x: x[0],
    )
    best = scored[0][1]
    alts = [
        {**h.to_dict(), "fusion_score": round(s, 4)}
        for s, h in scored[:4]
    ]
    return best, alts, neck


def _resolve_finger(
    hyp: StringHypothesis,
    active: list[int],
) -> dict[str, Any]:
    """String-aware fingering label constrained by pitch + curl proxies."""
    suggested = int(hyp.suggested_finger)
    if suggested == 0:
        if not active:
            return {
                "finger": 0,
                "label": "open",
                "confidence": 0.7,
                "source": "pitch_open + no_curl",
            }
        return {
            "finger": 0,
            "label": "open_preferred",
            "confidence": 0.45,
            "source": "pitch_open; curl ignored (open-string MIDI)",
            "curl_active": active,
        }

    if suggested in active:
        return {
            "finger": suggested,
            "label": f"finger_{suggested}",
            "confidence": 0.65,
            "source": "pitch_position ∩ curl",
            "curl_active": active,
        }
    if active:
        # Prefer highest active finger as sounding stop when ambiguous.
        sounding = max(active)
        return {
            "finger": sounding,
            "label": f"finger_{sounding}",
            "confidence": 0.4,
            "source": "curl_proxy; pitch suggested different stop",
            "pitch_suggested_finger": suggested,
            "curl_active": active,
        }
    return {
        "finger": suggested,
        "label": f"finger_{suggested}_pitch_only",
        "confidence": 0.35,
        "source": "pitch_position; no curl detected",
    }


def fuse_timeline(
    timeline: dict[str, Any],
    *,
    audio_notes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach string-aware fingering + bow-string labels to a video timeline.

    Expects ``audio_notes`` (or ``timeline['audio_notes']``) from the existing
    pyin pipeline. Vision-only fields are preserved; fusion adds parallel
    ``fused`` blocks without inventing concert-grade fingerings.
    """
    notes = audio_notes if audio_notes is not None else list(timeline.get("audio_notes") or [])
    samples = list(timeline.get("samples") or [])
    events = list(timeline.get("events") or [])

    fused_samples: list[dict[str, Any]] = []
    fused_events: list[dict[str, Any]] = []
    prev_key: tuple[Any, ...] | None = None

    for sample in samples:
        t = float(sample["t"])
        note = _note_at_time(notes, t)
        entry: dict[str, Any] = {
            "t": sample["t"],
            "frame_idx": sample.get("frame_idx"),
            "active_fingers": sample.get("active_fingers") or [],
            "bow_vision": sample.get("bow"),
        }
        if note is None:
            entry["pitch"] = None
            entry["string"] = None
            entry["fingering"] = None
            entry["bow"] = (
                {**(sample.get("bow") or {}), "string": None}
                if sample.get("bow")
                else None
            )
            fused_samples.append(entry)
            continue

        midi = int(note["midi"])
        hyp, alts, neck = _choose_hypothesis(midi, sample)
        fingering = (
            _resolve_finger(hyp, list(sample.get("active_fingers") or []))
            if hyp is not None
            else None
        )
        string_name = hyp.string if hyp else None
        bow = None
        if sample.get("bow"):
            bow = {
                **sample["bow"],
                "string": string_name,
                "string_source": "pyin_pitch_hypothesis",
            }
        elif string_name is not None:
            bow = {
                "direction": "unknown",
                "string": string_name,
                "string_source": "pyin_pitch_hypothesis",
            }

        entry.update(
            {
                "pitch": {
                    "midi": midi,
                    "note": note.get("note"),
                    "hz": note.get("pitch_hz"),
                    "confidence": note.get("confidence"),
                    "onset": note.get("onset"),
                    "offset": note.get("offset"),
                },
                "string": string_name,
                "position_band": hyp.position_band if hyp else None,
                "semitones_above_open": hyp.semitones_above_open if hyp else None,
                "hypotheses": alts,
                "neck_proxy": neck,
                "fingering": fingering,
                "bow": bow,
            }
        )
        fused_samples.append(entry)

        key = (
            string_name,
            (fingering or {}).get("finger"),
            (bow or {}).get("direction"),
        )
        if key != prev_key and string_name is not None:
            fused_events.append(
                {
                    "t": round(t, 3),
                    "type": "fused_note",
                    "string": string_name,
                    "finger": (fingering or {}).get("finger"),
                    "fingering_label": (fingering or {}).get("label"),
                    "midi": midi,
                    "note": note.get("note"),
                    "bow_direction": (bow or {}).get("direction"),
                    "confidence": round(
                        0.5
                        * float(note.get("confidence") or 0.5)
                        + 0.5 * float((fingering or {}).get("confidence") or 0.3),
                        3,
                    ),
                    "note_detail": "pitch→string + landmark-constrained finger; bow string from pitch",
                }
            )
            prev_key = key

    # Enrich original video events with nearest pitch string when possible.
    enriched_events: list[dict[str, Any]] = []
    for ev in events:
        e = dict(ev)
        note = _note_at_time(notes, float(ev["t"]))
        if note is not None:
            hyp = primary_hypothesis(int(note["midi"]))
            if hyp is not None:
                e["sounding_string"] = hyp.string
                e["sounding_midi"] = hyp.midi
                e["sounding_note"] = note.get("note")
                if e.get("type") == "bow":
                    e["string"] = hyp.string
        enriched_events.append(e)

    # Per-note fusion summary (one row per pyin note).
    note_rows: list[dict[str, Any]] = []
    for n in notes:
        onset = float(n["onset"])
        mid_t = onset + 0.5 * float(n.get("duration") or max(float(n["offset"]) - onset, 0.01))
        # Sample nearest fused frame at note midpoint.
        nearest = min(fused_samples, key=lambda s: abs(float(s["t"]) - mid_t), default=None)
        midi = int(n["midi"])
        hyp = primary_hypothesis(midi)
        row = {
            **{k: n[k] for k in n if k != "nearby_video_events"},
            "string": (nearest or {}).get("string") or (hyp.string if hyp else None),
            "fingering": (nearest or {}).get("fingering"),
            "position_band": (nearest or {}).get("position_band"),
            "semitones_above_open": (nearest or {}).get("semitones_above_open"),
            "hypotheses": (nearest or {}).get("hypotheses")
            or [h.to_dict() for h in pitch_string_hypotheses(midi)[:3]],
            "bow_string": ((nearest or {}).get("bow") or {}).get("string"),
            "nearby_video_events": n.get("nearby_video_events"),
        }
        note_rows.append(row)

    out = dict(timeline)
    out["schema"] = "cello_av_fusion_v0"
    out["fusion"] = {
        "ok": True,
        "method": "pyin pitch → open-string/position hypotheses ∩ left-hand curl/neck proxy; bow string from pitch",
        "strings": STRING_ORDER,
        "num_audio_notes": len(notes),
        "num_fused_events": len(fused_events),
        "limits": [
            "Hackathon fusion demo — not concert fingerings.",
            "String ID is the lowest-position pitch hypothesis re-ranked by hand height/curl.",
            "Finger indices remain curl proxies constrained by sounding pitch.",
            "Bow direction stays wrist-velocity; sounding string comes from pitch, not stick segmentation.",
        ],
    }
    out["events"] = enriched_events
    out["fused_events"] = fused_events
    out["fused_samples"] = fused_samples
    out["fused_notes"] = note_rows
    out["limits"] = list(timeline.get("limits") or []) + list(out["fusion"]["limits"])
    return out


def fusion_overlay_label(fused_sample: dict[str, Any] | None) -> str:
    """Short HUD string for overlay frames."""
    if not fused_sample:
        return ""
    parts: list[str] = []
    pitch = fused_sample.get("pitch") or {}
    if pitch.get("note"):
        parts.append(str(pitch["note"]))
    if fused_sample.get("string"):
        parts.append(f"str={fused_sample['string']}")
    fing = fused_sample.get("fingering") or {}
    if fing.get("label"):
        parts.append(str(fing["label"]))
    bow = fused_sample.get("bow") or {}
    if bow.get("direction") and bow["direction"] != "unknown":
        parts.append(f"bow={bow['direction']}")
        if bow.get("string"):
            parts.append(f"on {bow['string']}")
    return "  ".join(parts)
