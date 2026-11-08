"""Load the local public-domain cello score catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo root: .../src/cello_note_detector/identify/catalog.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_PATH = _REPO_ROOT / "data" / "catalog" / "catalog.json"


@dataclass(frozen=True)
class CatalogEntry:
    """One known piece in the local catalog."""

    id: str
    title: str
    edition: str
    reference_json: Path
    composer: str = ""
    license: str = "Public Domain"
    imslp_work: str | None = None
    key: str | None = None
    time_signature: str | None = None
    midi_path: Path | None = None
    min_confidence: float = 0.55
    match_notes: str = ""
    midi_sequence: list[int] = field(default_factory=list)
    reference: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_notes(self) -> int:
        return len(self.midi_sequence)


def _resolve_repo_path(raw: str | Path, *, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_catalog(
    catalog_path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> list[CatalogEntry]:
    """Load catalog entries and their monophonic reference sequences.

    Missing reference files are skipped with a warning so a partial catalog
    still works. An empty / missing catalog returns ``[]`` (callers fall back
    to transcription).
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG_PATH
    if not path.is_file():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    default_min = float(data.get("default_min_confidence", 0.55))
    entries: list[CatalogEntry] = []
    for raw in data.get("entries") or []:
        ref_path = _resolve_repo_path(raw["reference_json"], repo_root=root)
        if not ref_path.is_file():
            print(f"  warning: catalog entry {raw.get('id')!r} missing reference: {ref_path}")
            continue
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        seq = ref.get("midi_sequence")
        if not seq and ref.get("events"):
            seq = [int(e["midi"]) for e in ref["events"] if "midi" in e]
        if not seq:
            print(f"  warning: catalog entry {raw.get('id')!r} has no midi_sequence")
            continue
        midi_raw = raw.get("midi_path")
        midi_path = (
            _resolve_repo_path(midi_raw, repo_root=root) if midi_raw else None
        )
        entries.append(
            CatalogEntry(
                id=str(raw["id"]),
                title=str(raw.get("title") or ref.get("title") or raw["id"]),
                edition=str(
                    raw.get("edition")
                    or ref.get("source_edition")
                    or "Public Domain"
                ),
                reference_json=ref_path,
                composer=str(raw.get("composer") or ""),
                license=str(raw.get("license") or "Public Domain"),
                imslp_work=raw.get("imslp_work") or ref.get("imslp_work"),
                key=raw.get("key") or ref.get("key"),
                time_signature=raw.get("time_signature") or ref.get("time_signature"),
                midi_path=midi_path,
                min_confidence=float(raw.get("min_confidence", default_min)),
                match_notes=str(raw.get("match_notes") or ""),
                midi_sequence=[int(x) for x in seq],
                reference=ref,
            )
        )
    return entries
