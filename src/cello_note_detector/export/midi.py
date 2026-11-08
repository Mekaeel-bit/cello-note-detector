"""Minimal Type-1 MIDI export from quantized note/rest timelines.

No third-party MIDI library — writes Standard MIDI File bytes directly.
Rests become silence (no MIDI events); tempo comes from the rhythm layer.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

# Pulses (ticks) per quarter note.
PPQ = 480
DEFAULT_VELOCITY = 80
DEFAULT_CHANNEL = 0


def _encode_vlq(value: int) -> bytes:
    """MIDI variable-length quantity."""
    value = int(value)
    if value < 0:
        raise ValueError("VLQ cannot encode negative values")
    buffer = value & 0x7F
    out = bytearray()
    while value >> 7:
        value >>= 7
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def _beats_to_ticks(beats: float, *, ppq: int = PPQ) -> int:
    return max(0, int(round(float(beats) * ppq)))


def _iter_note_events(result: dict) -> list[dict]:
    """Sounding notes only (rests implied by gaps / sequential timing)."""
    events = result.get("events")
    if events:
        return [e for e in events if e.get("type") == "note"]
    return list(result.get("notes") or [])


def _bpm_from_result(result: dict, default: float = 120.0) -> float:
    tempo = result.get("tempo")
    if isinstance(tempo, dict) and tempo.get("bpm"):
        return float(tempo["bpm"])
    return default


def build_midi_track_events(
    result: dict,
    *,
    ppq: int = PPQ,
    velocity: int = DEFAULT_VELOCITY,
    channel: int = DEFAULT_CHANNEL,
    sequential: bool = True,
) -> list[tuple[int, bytes]]:
    """Build absolute-tick MIDI events for one track.

    When ``sequential`` is True (default), notes are placed back-to-back using
    quantized durations and the full chronological ``events`` timeline (so
    rests become ticks of silence). When False, uses absolute
    ``quantized_onset_beats`` on notes only.
    """
    bpm = _bpm_from_result(result)
    # Microseconds per quarter note.
    tempo_us = int(round(60_000_000 / max(bpm, 1e-6)))

    events_abs: list[tuple[int, bytes]] = []
    # Meta: track name + tempo + time signature 4/4.
    events_abs.append((0, b"\xff\x03" + _encode_vlq(5) + b"Cello"))
    events_abs.append(
        (0, b"\xff\x51\x03" + struct.pack(">I", tempo_us)[1:])
    )
    events_abs.append((0, b"\xff\x58\x04\x04\x02\x18\x08"))  # 4/4

    if sequential:
        timeline = result.get("events")
        if not timeline:
            # Synthesize timeline from notes + rests.
            timeline = []
            for n in result.get("notes") or []:
                item = dict(n)
                item["type"] = "note"
                timeline.append(item)
            for r in result.get("rests") or []:
                item = dict(r)
                item["type"] = "rest"
                timeline.append(item)
            timeline.sort(
                key=lambda e: (
                    float(e.get("onset", 0.0)),
                    0 if e.get("type") == "note" else 1,
                )
            )

        tick = 0
        for item in timeline:
            beats = float(
                item.get("duration_value_beats")
                or item.get("quantized_duration_beats")
                or item.get("duration_beats")
                or 0.0
            )
            dur_ticks = _beats_to_ticks(beats, ppq=ppq)
            if item.get("type") == "rest" or "midi" not in item:
                tick += dur_ticks
                continue
            midi = int(item["midi"])
            vel = max(1, min(127, int(velocity)))
            ch = channel & 0x0F
            events_abs.append((tick, bytes([0x90 | ch, midi & 0x7F, vel])))
            events_abs.append(
                (tick + dur_ticks, bytes([0x80 | ch, midi & 0x7F, 0x40]))
            )
            tick += dur_ticks
    else:
        for note in _iter_note_events(result):
            onset_beats = float(
                note.get("quantized_onset_beats")
                or note.get("onset_beats")
                or 0.0
            )
            dur_beats = float(
                note.get("duration_value_beats")
                or note.get("quantized_duration_beats")
                or note.get("duration_beats")
                or 1.0
            )
            start = _beats_to_ticks(onset_beats, ppq=ppq)
            end = start + _beats_to_ticks(dur_beats, ppq=ppq)
            midi = int(note["midi"])
            vel = max(1, min(127, int(velocity)))
            ch = channel & 0x0F
            events_abs.append((start, bytes([0x90 | ch, midi & 0x7F, vel])))
            events_abs.append((end, bytes([0x80 | ch, midi & 0x7F, 0x40])))

    # End of track.
    end_tick = max((t for t, _ in events_abs), default=0)
    events_abs.append((end_tick, b"\xff\x2f\x00"))
    events_abs.sort(key=lambda pair: (pair[0], pair[1][0] & 0xF0))
    return events_abs


def _write_track(fh: BinaryIO, abs_events: list[tuple[int, bytes]]) -> None:
    body = bytearray()
    last_tick = 0
    for tick, payload in abs_events:
        delta = max(0, tick - last_tick)
        body.extend(_encode_vlq(delta))
        body.extend(payload)
        last_tick = tick
    fh.write(b"MTrk")
    fh.write(struct.pack(">I", len(body)))
    fh.write(body)


def write_midi(
    result: dict,
    path: str | Path,
    *,
    ppq: int = PPQ,
    velocity: int = DEFAULT_VELOCITY,
) -> Path:
    """Write a Type-1 SMF (one conductor-ish track) for ``result``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    abs_events = build_midi_track_events(
        result, ppq=ppq, velocity=velocity, sequential=True
    )
    with out.open("wb") as fh:
        # Header: format 1, 1 track, PPQ division.
        fh.write(b"MThd")
        fh.write(struct.pack(">IHHH", 6, 1, 1, ppq))
        _write_track(fh, abs_events)
    return out
