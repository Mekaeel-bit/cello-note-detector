"""Note-event types and serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NoteEvent:
    """A contiguous voiced segment collapsed to one note event."""

    pitch_hz: float
    midi: int
    note: str
    onset: float
    offset: float
    duration: float
    confidence: float


def note_events_to_dicts(events: list[NoteEvent]) -> list[dict]:
    """Serialize note events for JSON."""
    return [asdict(e) for e in events]
