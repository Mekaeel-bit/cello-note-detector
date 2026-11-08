"""Load the extensible local public-domain cello score catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .musicxml_edition import parse_musicxml_fingerprint

# Repo root: .../src/cello_note_detector/identify/catalog.py → parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_PATH = _REPO_ROOT / "data" / "catalog" / "catalog.json"


@dataclass(frozen=True)
class CatalogEntry:
    """One known piece (or movement) in the local PD catalog."""

    id: str
    title: str
    edition: str
    edition_musicxml: Path
    composer: str = ""
    license: str = "Public Domain"
    imslp_work: str | None = None
    key: str | None = None
    time_signature: str | None = None
    midi_path: Path | None = None
    fingerprint_json: Path | None = None
    min_confidence: float = 0.55
    match_notes: str = ""
    typical_duration_sec: float | None = None
    midi_sequence: list[int] = field(default_factory=list)
    ornaments: dict[str, Any] = field(default_factory=dict)
    # Legacy eval reference (optional; matching prefers MusicXML fingerprint).
    reference_json: Path | None = None
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
    """Load catalog entries from MusicXML editions + fingerprints.

    Designed so any PD IMSLP/Mutopia/CC0 MusicXML can be added via
    ``scripts/add_catalog_piece.py``. Missing edition files are skipped with a
    warning; an empty catalog returns ``[]`` (callers fall back to transcription).
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG_PATH
    if not path.is_file():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    default_min = float(data.get("default_min_confidence", 0.55))
    entries: list[CatalogEntry] = []
    for raw in data.get("entries") or []:
        edition_raw = raw.get("edition_musicxml") or raw.get("musicxml")
        if not edition_raw:
            print(
                f"  warning: catalog entry {raw.get('id')!r} missing "
                "edition_musicxml — skip"
            )
            continue
        edition_path = _resolve_repo_path(edition_raw, repo_root=root)
        if not edition_path.is_file():
            print(
                f"  warning: catalog entry {raw.get('id')!r} missing edition: "
                f"{edition_path}"
            )
            continue

        # Prefer committed fingerprint; else extract from the edition file.
        fp: dict[str, Any] = {}
        fp_path = None
        if raw.get("fingerprint_json"):
            fp_path = _resolve_repo_path(raw["fingerprint_json"], repo_root=root)
            if fp_path.is_file():
                fp = json.loads(fp_path.read_text(encoding="utf-8"))
        if not fp.get("midi_sequence"):
            try:
                fp = parse_musicxml_fingerprint(edition_path)
            except Exception as exc:  # noqa: BLE001 — keep catalog robust
                print(
                    f"  warning: catalog entry {raw.get('id')!r} fingerprint "
                    f"failed: {exc}"
                )
                continue

        seq = [int(x) for x in fp["midi_sequence"]]
        if not seq:
            print(f"  warning: catalog entry {raw.get('id')!r} empty fingerprint")
            continue

        midi_raw = raw.get("midi_path")
        midi_path = (
            _resolve_repo_path(midi_raw, repo_root=root) if midi_raw else None
        )
        ref_path = None
        ref: dict[str, Any] = {}
        if raw.get("reference_json"):
            ref_path = _resolve_repo_path(raw["reference_json"], repo_root=root)
            if ref_path.is_file():
                ref = json.loads(ref_path.read_text(encoding="utf-8"))

        entries.append(
            CatalogEntry(
                id=str(raw["id"]),
                title=str(
                    raw.get("title") or fp.get("title") or raw["id"]
                ),
                edition=str(raw.get("edition") or "Public Domain"),
                edition_musicxml=edition_path,
                composer=str(raw.get("composer") or ""),
                license=str(raw.get("license") or "Public Domain"),
                imslp_work=raw.get("imslp_work"),
                key=raw.get("key"),
                time_signature=raw.get("time_signature"),
                midi_path=midi_path,
                fingerprint_json=fp_path,
                min_confidence=float(raw.get("min_confidence", default_min)),
                match_notes=str(raw.get("match_notes") or ""),
                typical_duration_sec=(
                    float(raw["typical_duration_sec"])
                    if raw.get("typical_duration_sec") is not None
                    else None
                ),
                midi_sequence=seq,
                ornaments=dict(fp.get("ornaments") or {}),
                reference_json=ref_path,
                reference=ref,
            )
        )
    return entries


def search_catalog(
    query: str,
    catalog_path: str | Path | None = None,
    *,
    repo_root: Path | None = None,
    entries: list[CatalogEntry] | None = None,
) -> list[CatalogEntry]:
    """Filter catalog entries by substring match on id/title/composer/edition.

    This is the human-facing PD lookup over the local extensible index.
    Motif matching for audio ID lives in ``match.identify_sequence``.
    Empty ``query`` returns every loaded entry.
    """
    loaded = (
        list(entries)
        if entries is not None
        else load_catalog(catalog_path, repo_root=repo_root)
    )
    q = (query or "").strip().lower()
    if not q:
        return loaded
    hits: list[CatalogEntry] = []
    for entry in loaded:
        hay = " ".join(
            [
                entry.id,
                entry.title,
                entry.composer,
                entry.edition,
                entry.license,
                entry.imslp_work or "",
                entry.key or "",
            ]
        ).lower()
        if q in hay:
            hits.append(entry)
    return hits


def format_catalog_listing(entries: list[CatalogEntry]) -> str:
    """Pretty-print catalog entries for CLI / docs."""
    if not entries:
        return "(no catalog entries)"
    lines: list[str] = []
    for e in entries:
        dur = (
            f", ~{e.typical_duration_sec:.0f}s"
            if e.typical_duration_sec is not None
            else ""
        )
        lines.append(
            f"{e.id}: {e.title} — {e.composer} [{e.license}] "
            f"({e.num_notes} notes{dur})"
        )
        lines.append(f"  edition: {e.edition_musicxml}")
        if e.imslp_work:
            lines.append(f"  imslp: {e.imslp_work}")
    return "\n".join(lines)
