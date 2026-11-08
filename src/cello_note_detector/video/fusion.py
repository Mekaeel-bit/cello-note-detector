"""Fuse pyin pitch with video landmarks → string-aware fingering + bow string.

Whole-clip tracking: sticky left-hand position from neck height (with hysteresis),
string continuity across notes, and fingering from (position, string, pitch) —
not curl proxies. Curl is diagnostic only.
"""

from __future__ import annotations

from typing import Any

from cello_note_detector.video.strings import (
    OPEN_MIDI_BY_STRING,
    POSITION_BASE_ST,
    POSITION_SEQUENCE,
    STRING_ORDER,
    finger_for_stop,
    pitch_string_hypotheses,
    position_needed_for_stop,
    primary_hypothesis,
)


def _note_at_time(notes: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for n in notes:
        onset = float(n.get("onset", 0.0))
        offset = float(n.get("offset", onset))
        if onset <= t < offset:
            if best is None or onset >= float(best.get("onset", -1)):
                best = n
    return best


def _hand_neck_proxy(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Fingerboard-height proxy from left-hand landmarks (front camera).

    Smaller image-y ≈ nut/scroll; larger y ≈ bridge. ``neck_height`` ∈ [0, 1].
    """
    left = sample.get("left_hand")
    if not left:
        return None
    wrist = left.get("wrist") or {}
    tips = left.get("tips") or {}
    ys: list[float] = []
    if "y" in wrist:
        ys.append(float(wrist["y"]))
    for name in ("index", "middle", "ring", "pinky"):
        tip = tips.get(name)
        if tip and "y" in tip:
            ys.append(float(tip["y"]))
    if not ys:
        return None
    y_mean = sum(ys) / len(ys)
    nut, bridge = 0.12, 0.55
    height = (y_mean - nut) / max(bridge - nut, 1e-3)
    height = max(0.0, min(1.0, height))
    return {
        "y_mean": round(y_mean, 4),
        "neck_height": round(height, 4),
        "band": _height_to_position(height),
    }


def _height_to_position(height: float) -> str:
    """Map normalized neck height → sticky LH position label."""
    if height < 0.28:
        return "I"
    if height < 0.42:
        return "II"
    if height < 0.55:
        return "III"
    if height < 0.70:
        return "IV"
    return "thumb"


def _neck_timeline(samples: list[dict[str, Any]]) -> list[tuple[float, str, float]]:
    """Return [(t, position_label, neck_height), ...] for frames with a left hand."""
    out: list[tuple[float, str, float]] = []
    for sample in samples:
        neck = _hand_neck_proxy(sample)
        if neck is None:
            continue
        out.append((float(sample["t"]), str(neck["band"]), float(neck["neck_height"])))
    return out


def _neck_at(
    timeline: list[tuple[float, str, float]],
    t: float,
) -> tuple[str | None, float | None]:
    if not timeline:
        return None, None
    # Nearest sample in time (vision may end before audio).
    best = min(timeline, key=lambda x: abs(x[0] - t))
    return best[1], best[2]


def _pos_index(label: str) -> int:
    if label in POSITION_SEQUENCE:
        return POSITION_SEQUENCE.index(label)
    if label == "high":
        return len(POSITION_SEQUENCE) - 1
    return 0


def _bump_position(current: str, target: str) -> str:
    """Move sticky position at most one step toward ``target`` (smooth shifts)."""
    if current == target:
        return current
    ci, ti = _pos_index(current), _pos_index(target)
    if ti > ci:
        return POSITION_SEQUENCE[min(ci + 1, len(POSITION_SEQUENCE) - 1)]
    if ti < ci:
        return POSITION_SEQUENCE[max(ci - 1, 0)]
    return current


def _reachable(midi: int, string: str, position: str) -> tuple[bool, int | None, int]:
    open_midi = OPEN_MIDI_BY_STRING[string]
    st = int(midi) - open_midi
    if st < 0 or st > 24:
        return False, None, st
    finger = finger_for_stop(st, position=position)
    return finger is not None, finger, st


def _choose_string_and_position(
    midi: int,
    *,
    sticky_string: str | None,
    sticky_position: str,
    vision_position: str | None,
) -> tuple[str, str, int, int, list[dict[str, Any]]]:
    """Pick string + LH position for ``midi`` with continuity priors.

    Prefers continuing ``sticky_string`` / ``sticky_position`` when the pitch
    still fits. Never labels a non-open pitch as finger 0. Vision suggests
    shifts but does not yank the hand on every frame.
    """
    hyps = pitch_string_hypotheses(midi)
    if not hyps:
        # Fallback — should be rare inside cello range.
        return sticky_string or "A", sticky_position, 0, 0, []

    # Desired position: stay sticky. Vision may nudge one step. Pitch reach
    # (below) is the main reason to climb — keeps early first-position bars stable.
    desired = sticky_position
    if vision_position is not None and vision_position != sticky_position:
        if abs(_pos_index(vision_position) - _pos_index(sticky_position)) >= 1:
            desired = _bump_position(sticky_position, vision_position)

    def reachable_on_preferred(pos: str) -> bool:
        order = []
        if sticky_string:
            order.append(sticky_string)
        order.extend(h.string for h in hyps if h.string not in order)
        return any(_reachable(midi, s, pos)[0] for s in order)

    # If continuing on a string whose stop sits well below the sticky seat
    # (descending after a high climb), ease the hand down one step at a time.
    if sticky_string and sticky_string in OPEN_MIDI_BY_STRING:
        st_here = midi - OPEN_MIDI_BY_STRING[sticky_string]
        if 0 <= st_here < POSITION_BASE_ST.get(desired, 0):
            desired = _bump_position(desired, position_needed_for_stop(st_here))

    # Pitch-forced climb when the seat cannot reach the note.
    guard = 0
    while not reachable_on_preferred(desired) and guard < 8:
        min_st = min(h.semitones_above_open for h in hyps)
        need = position_needed_for_stop(min_st)
        if sticky_string and sticky_string in OPEN_MIDI_BY_STRING:
            st_here = midi - OPEN_MIDI_BY_STRING[sticky_string]
            if st_here >= 0:
                need = position_needed_for_stop(st_here)
        desired = _bump_position(desired, need)
        guard += 1

    scored: list[tuple[float, str, str, int, int]] = []
    alts: list[dict[str, Any]] = []
    for h in hyps:
        ok, finger, st = _reachable(midi, h.string, desired)
        if not ok or finger is None:
            # Try the minimal position that works on this string alone.
            need = position_needed_for_stop(st)
            ok2, finger2, _ = _reachable(midi, h.string, need)
            if not ok2 or finger2 is None:
                continue
            pos_used, finger_used = need, finger2
            shift_pen = 2.5 + abs(_pos_index(need) - _pos_index(desired))
        else:
            pos_used, finger_used = desired, finger
            shift_pen = 0.0

        score = float(st) * 0.15 + shift_pen
        # Strong continuity: stay on the same string when playable.
        if sticky_string and h.string == sticky_string:
            score -= 4.0
        # Mild preference for not leaping many strings.
        if sticky_string:
            score += 0.4 * abs(
                STRING_ORDER.index(h.string) - STRING_ORDER.index(sticky_string)
            )
        # Prefer staying in sticky / desired position.
        score += 0.5 * abs(_pos_index(pos_used) - _pos_index(desired))
        # Open string only when pitch matches open (finger_used == 0).
        if finger_used == 0 and st != 0:
            score += 50.0  # hard reject
        scored.append((score, h.string, pos_used, finger_used, st))
        alts.append(
            {
                **h.to_dict(),
                "fusion_score": round(score, 4),
                "position": pos_used,
                "finger": finger_used,
            }
        )

    if not scored:
        h0 = hyps[0]
        st = h0.semitones_above_open
        pos = position_needed_for_stop(st)
        finger = finger_for_stop(st, position=pos) or 4
        return h0.string, pos, finger, st, alts

    scored.sort(key=lambda x: (x[0], x[4]))
    _score, string, pos, finger, st = scored[0]
    alts.sort(key=lambda a: a["fusion_score"])
    return string, pos, int(finger), int(st), alts[:4]


def _assign_notes(
    notes: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Chronological whole-clip assignment with sticky string/position."""
    neck_tl = _neck_timeline(samples)
    sticky_string: str | None = None
    sticky_position = "I"
    sticky_height: float | None = None
    # Vision must move neck_height by this much vs the height at last settle
    # before we allow a vision-suggested shift (noise rejection).
    vision_shift_eps = 0.14
    pending_vision: str | None = None
    pending_count = 0

    rows: list[dict[str, Any]] = []
    for n in sorted(notes, key=lambda x: float(x.get("onset", 0.0))):
        midi = int(n["midi"])
        onset = float(n["onset"])
        mid_t = onset + 0.5 * float(
            n.get("duration") or max(float(n["offset"]) - onset, 0.01)
        )
        vision_pos, neck_h = _neck_at(neck_tl, mid_t)

        # Vision may suggest a shift only after a clear neck-height move.
        vision_for_choose = None
        if vision_pos is not None and neck_h is not None:
            if sticky_height is None:
                sticky_height = neck_h
            height_delta = abs(neck_h - sticky_height)
            if vision_pos == sticky_position or height_delta < vision_shift_eps:
                pending_vision, pending_count = None, 0
            elif vision_pos == pending_vision:
                pending_count += 1
                if pending_count >= 3:
                    vision_for_choose = vision_pos
                    pending_vision, pending_count = None, 0
            else:
                pending_vision, pending_count = vision_pos, 1

        string, position, finger, st, alts = _choose_string_and_position(
            midi,
            sticky_string=sticky_string,
            sticky_position=sticky_position,
            vision_position=vision_for_choose,
        )
        if position != sticky_position and neck_h is not None:
            sticky_height = neck_h
        sticky_string = string
        sticky_position = position

        fingering = {
            "finger": finger,
            "label": "open" if finger == 0 else f"finger_{finger}",
            "confidence": 0.7 if finger == 0 or st <= 8 else 0.55,
            "source": "position+string+pitch",
            "lh_position": position,
        }
        row = {
            **{k: n[k] for k in n if k != "nearby_video_events"},
            "string": string,
            "fingering": fingering,
            "lh_position": position,
            "position_band": (
                "open"
                if st == 0
                else ("low" if st <= 4 else ("mid" if st <= 8 else "high"))
            ),
            "semitones_above_open": st,
            "position_base_st": POSITION_BASE_ST.get(position, 0),
            "hypotheses": alts,
            "bow_string": string,
            "neck_height": neck_h,
            "nearby_video_events": n.get("nearby_video_events"),
        }
        rows.append(row)
    return rows


def fuse_timeline(
    timeline: dict[str, Any],
    *,
    audio_notes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach string-aware fingering + bow-string labels to a video timeline.

    Processes **all** pyin notes for the clip (not capped by ``--max-frames``).
    Video landmarks update sticky LH position while available; after vision
    ends, pitch continuity keeps string/position tracking.
    """
    notes = audio_notes if audio_notes is not None else list(timeline.get("audio_notes") or [])
    samples = list(timeline.get("samples") or [])
    events = list(timeline.get("events") or [])

    note_rows = _assign_notes(notes, samples)
    # Index for sample overlay lookup.
    note_by_mid: list[tuple[float, dict[str, Any]]] = []
    for row in note_rows:
        onset = float(row["onset"])
        mid = onset + 0.5 * float(
            row.get("duration") or max(float(row["offset"]) - onset, 0.01)
        )
        note_by_mid.append((mid, row))

    def row_at(t: float) -> dict[str, Any] | None:
        if not note_by_mid:
            return None
        return min(note_by_mid, key=lambda x: abs(x[0] - t))[1]

    fused_samples: list[dict[str, Any]] = []
    fused_events: list[dict[str, Any]] = []
    prev_key: tuple[Any, ...] | None = None

    for sample in samples:
        t = float(sample["t"])
        row = row_at(t)
        note = _note_at_time(notes, t)
        neck = _hand_neck_proxy(sample)
        entry: dict[str, Any] = {
            "t": sample["t"],
            "frame_idx": sample.get("frame_idx"),
            "active_fingers": sample.get("active_fingers") or [],
            "bow_vision": sample.get("bow"),
            "neck_proxy": neck,
        }
        if note is None or row is None:
            entry.update(
                {
                    "pitch": None,
                    "string": None,
                    "fingering": None,
                    "lh_position": None,
                    "bow": (
                        {**(sample.get("bow") or {}), "string": None}
                        if sample.get("bow")
                        else None
                    ),
                }
            )
            fused_samples.append(entry)
            continue

        string_name = row["string"]
        fingering = row["fingering"]
        bow = None
        if sample.get("bow"):
            bow = {
                **sample["bow"],
                "string": string_name,
                "string_source": "pyin_pitch_hypothesis",
            }
        else:
            bow = {
                "direction": "unknown",
                "string": string_name,
                "string_source": "pyin_pitch_hypothesis",
            }
        entry.update(
            {
                "pitch": {
                    "midi": int(note["midi"]),
                    "note": note.get("note"),
                    "hz": note.get("pitch_hz"),
                    "confidence": note.get("confidence"),
                    "onset": note.get("onset"),
                    "offset": note.get("offset"),
                },
                "string": string_name,
                "lh_position": row.get("lh_position"),
                "position_band": row.get("position_band"),
                "semitones_above_open": row.get("semitones_above_open"),
                "hypotheses": row.get("hypotheses"),
                "fingering": fingering,
                "bow": bow,
            }
        )
        fused_samples.append(entry)

        key = (string_name, fingering.get("finger"), row.get("lh_position"))
        if key != prev_key and string_name is not None:
            fused_events.append(
                {
                    "t": round(t, 3),
                    "type": "fused_note",
                    "string": string_name,
                    "finger": fingering.get("finger"),
                    "fingering_label": fingering.get("label"),
                    "lh_position": row.get("lh_position"),
                    "midi": int(note["midi"]),
                    "note": note.get("note"),
                    "bow_direction": bow.get("direction"),
                    "confidence": fingering.get("confidence"),
                    "note_detail": "sticky LH position + string continuity; finger from pitch",
                }
            )
            prev_key = key

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

    out = dict(timeline)
    out["schema"] = "cello_av_fusion_v1"
    out["fusion"] = {
        "ok": True,
        "method": (
            "sticky LH position (neck height + pitch reach) → string continuity → "
            "finger = f(position, string, pitch); curl proxies not used for engraving"
        ),
        "strings": STRING_ORDER,
        "num_audio_notes": len(notes),
        "num_fused_events": len(fused_events),
        "limits": [
            "Hackathon fusion — approximate fingerings/positions for monophonic cello.",
            "Position is sticky from hand height with hysteresis; pitch forces shifts when unreachable.",
            "Fingering comes from position+string+pitch (not MediaPipe curl).",
            "Bow direction stays wrist-velocity; sounding string from pitch hypothesis.",
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
    if fused_sample.get("lh_position"):
        parts.append(f"pos={fused_sample['lh_position']}")
    fing = fused_sample.get("fingering") or {}
    if fing.get("label"):
        parts.append(str(fing["label"]))
    bow = fused_sample.get("bow") or {}
    if bow.get("direction") and bow["direction"] != "unknown":
        parts.append(f"bow={bow['direction']}")
        if bow.get("string"):
            parts.append(f"on {bow['string']}")
    return "  ".join(parts)
