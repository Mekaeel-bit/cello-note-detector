"""Shazam-like PD score discovery: local catalog first, then remote seeds.

Flow
----
1. Fingerprint audio (caller already has MIDI pitch sequence).
2. Match against the **local** extensible catalog (bundled PD MusicXML).
3. On miss, consult ``discovery_index.json`` for known PD works (IMSLP /
   Mutopia / CC mirrors) and attempt a **live** MusicXML/MXL/MIDI fetch when
   a machine-readable URL or IMSLP work page is listed.
4. Re-score against any successfully fetched edition; on miss → transcription.

Important UX
------------
Files under ``data/catalog/pieces/`` are **bundled PD editions** that ship with
the repo. Seeing them on disk before ``--identify`` does **not** mean a live
song ID already happened. A MATCH only occurs when identify copies an edition
into the output dir and records ``score_kind=verbatim_pd_edition``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .catalog import CatalogEntry, load_catalog
from .match import IdentificationResult, identify_sequence, score_against_entry
from .musicxml_edition import parse_musicxml_fingerprint

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DISCOVERY_PATH = _REPO_ROOT / "data" / "catalog" / "discovery_index.json"

_UA = (
    "Mozilla/5.0 (compatible; cello-note-detector/0.1; "
    "+https://github.com/Mekaeel-bit/cello-note-detector)"
)
_COOKIES = "imslpdisclaimeraccepted=yes; imslp_wikiLanguageSelectorLanguage=en"
_HOSTS = ("ks15", "ks7", "ks9", "s9", "ca1")


@dataclass
class DiscoveryCandidate:
    """One remote PD work that *might* have a fetchable MusicXML/MXL/MIDI."""

    id: str
    title: str
    composer: str
    license: str = "Public Domain"
    imslp_work: str | None = None
    musicxml_urls: list[str] = field(default_factory=list)
    midi_urls: list[str] = field(default_factory=list)
    imslp_mxl_id: str | None = None
    imslp_prefer: str = "mxl"
    key: str | None = None
    typical_duration_sec: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveryAttempt:
    """Result of one remote discovery attempt (success or explainable miss)."""

    candidate_id: str
    status: str
    detail: str
    fetched_path: str | None = None
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_discovery_index(
    path: str | Path | None = None,
) -> list[DiscoveryCandidate]:
    """Load remote PD discovery seeds (extensible; not limited to Bach/Gabrielli)."""
    p = Path(path) if path else DEFAULT_DISCOVERY_PATH
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    out: list[DiscoveryCandidate] = []
    for raw in data.get("candidates") or []:
        out.append(
            DiscoveryCandidate(
                id=str(raw["id"]),
                title=str(raw.get("title") or raw["id"]),
                composer=str(raw.get("composer") or ""),
                license=str(raw.get("license") or "Public Domain"),
                imslp_work=raw.get("imslp_work"),
                musicxml_urls=list(raw.get("musicxml_urls") or []),
                midi_urls=list(raw.get("midi_urls") or []),
                imslp_mxl_id=(
                    str(raw["imslp_mxl_id"]) if raw.get("imslp_mxl_id") else None
                ),
                imslp_prefer=str(raw.get("imslp_prefer") or "mxl"),
                key=raw.get("key"),
                typical_duration_sec=(
                    float(raw["typical_duration_sec"])
                    if raw.get("typical_duration_sec") is not None
                    else None
                ),
                notes=str(raw.get("notes") or ""),
            )
        )
    return out


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _UA, "Cookie": _COOKIES, "Referer": "https://imslp.org/"},
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def _looks_like_score_bytes(data: bytes) -> bool:
    if len(data) < 2000:
        return False
    head = data[:200].lstrip()
    if head.startswith(b"<") or head.startswith(b"<!DOCTYPE") or head.startswith(
        b"<html"
    ):
        # MusicXML is XML — allow if it looks like score-partwise / museScore
        low = data[:2000].lower()
        if b"score-partwise" in low or b"musescore" in low or b"midi" in low:
            return True
        if head.startswith(b"<!DOCTYPE html") or b"<html" in low[:100]:
            return False
    return True


def _download_url(url: str, dest: Path) -> Path:
    data = _http_get(url)
    if not _looks_like_score_bytes(data):
        raise RuntimeError(f"URL did not return a score payload: {url}")
    dest.write_bytes(data)
    return dest


def _fetch_imslp_from_work(
    work_url: str,
    *,
    prefer: str,
    imslp_id: str | None,
    dest: Path,
) -> tuple[Path, str]:
    """Minimal IMSLP work-page → CDN fetch (same pattern as fetch_imslp_score.py)."""
    html = _http_get(work_url).decode("utf-8", "replace")
    if imslp_id:
        try:
            rl = _http_get(
                f"https://imslp.org/wiki/Special:ReverseLookup/{imslp_id}"
            ).decode("utf-8", "replace")
            html += "\n" + rl
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: ReverseLookup failed: {exc}", file=sys.stderr)

    import re

    links: list[tuple[str, str, str]] = []
    for m in re.finditer(
        r"/images/([0-9a-f])/([0-9a-f]{2})/([^\"'<> ]+)", html, flags=re.I
    ):
        links.append((m.group(1), m.group(2), urllib.parse.unquote(m.group(3))))
    prefer = prefer.lower().lstrip(".")
    candidates = [c for c in links if c[2].lower().endswith("." + prefer)]
    if not candidates:
        raise RuntimeError(f"No .{prefer} files on IMSLP work page")

    chosen = candidates[0]
    if imslp_id:
        for c in candidates:
            if imslp_id in c[2] or f"IMSLP{imslp_id}" in c[2]:
                chosen = c
                break
    h1, h2, filename = chosen
    names = [urllib.parse.quote(filename, safe="().-_")]
    if imslp_id and not filename.upper().startswith("IMSLP"):
        names.insert(
            0,
            urllib.parse.quote(f"IMSLP{imslp_id}-{filename}", safe="().-_"),
        )

    last_err: Exception | None = None
    for host in _HOSTS:
        for name in names:
            url = f"https://{host}.imslp.org/files/imglnks/usimg/{h1}/{h2}/{name}"
            try:
                data = _http_get(url)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
            if not _looks_like_score_bytes(data):
                continue
            if data[:4] == b"%PDF" and prefer != "pdf":
                continue
            dest.write_bytes(data)
            return dest, url
    raise RuntimeError(f"IMSLP CDN fetch failed: {last_err}")


def fetch_discovery_candidate(
    candidate: DiscoveryCandidate,
    *,
    cache_dir: Path,
) -> DiscoveryAttempt:
    """Try to download a machine-readable PD score for ``candidate``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Direct MusicXML / MXL URLs
    for url in candidate.musicxml_urls:
        suffix = ".mxl" if url.lower().endswith(".mxl") else ".musicxml"
        dest = cache_dir / f"{candidate.id}{suffix}"
        try:
            _download_url(url, dest)
            return DiscoveryAttempt(
                candidate_id=candidate.id,
                status="fetched",
                detail="direct_musicxml_url",
                fetched_path=str(dest),
                source_url=url,
            )
        except Exception as exc:  # noqa: BLE001
            last = f"musicxml_url failed: {exc}"
    else:
        last = "no musicxml_urls"

    # 2) MIDI URLs (fingerprint / last-resort edition proxy)
    for url in candidate.midi_urls:
        dest = cache_dir / f"{candidate.id}.mid"
        try:
            _download_url(url, dest)
            return DiscoveryAttempt(
                candidate_id=candidate.id,
                status="fetched_midi_only",
                detail=(
                    "Got MIDI (fingerprint usable); MusicXML edition still missing "
                    "— attach requires MusicXML/MXL for verbatim PD export"
                ),
                fetched_path=str(dest),
                source_url=url,
            )
        except Exception as exc:  # noqa: BLE001
            last = f"midi_url failed: {exc}"

    # 3) IMSLP work page discovery
    if candidate.imslp_work:
        dest = cache_dir / f"{candidate.id}.{candidate.imslp_prefer}"
        try:
            path, url = _fetch_imslp_from_work(
                candidate.imslp_work,
                prefer=candidate.imslp_prefer,
                imslp_id=candidate.imslp_mxl_id,
                dest=dest,
            )
            return DiscoveryAttempt(
                candidate_id=candidate.id,
                status="fetched",
                detail="imslp_work_page",
                fetched_path=str(path),
                source_url=url,
            )
        except Exception as exc:  # noqa: BLE001
            last = f"imslp fetch failed: {exc}"
            return DiscoveryAttempt(
                candidate_id=candidate.id,
                status="unavailable",
                detail=(
                    f"{last}. Many IMSLP cello Caprices are PDF-only; "
                    "bundle MusicXML via scripts/add_catalog_piece.py when you "
                    "have a PD/CC edition."
                ),
            )

    return DiscoveryAttempt(
        candidate_id=candidate.id,
        status="unavailable",
        detail=last,
    )


def rank_discovery_candidates(
    candidates: Sequence[DiscoveryCandidate],
    *,
    duration_sec: float | None,
    inferred_key: str | None,
) -> list[DiscoveryCandidate]:
    """Prefer duration/key-aligned seeds (never uses filenames)."""
    scored: list[tuple[float, DiscoveryCandidate]] = []
    key_l = (inferred_key or "").lower().replace(" ", "")
    for c in candidates:
        score = 0.0
        if (
            duration_sec is not None
            and c.typical_duration_sec is not None
            and c.typical_duration_sec > 0
        ):
            rel = abs(duration_sec - c.typical_duration_sec) / c.typical_duration_sec
            score += max(0.0, 1.0 - rel)
        if key_l and c.key:
            ck = c.key.lower().replace(" ", "")
            if ck in key_l or key_l in ck:
                score += 0.5
            # rough tonic letter match (e.g. Gm vs G minor)
            if ck[:1] and ck[:1] == key_l[:1]:
                score += 0.15
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def entry_from_fetched_score(
    candidate: DiscoveryCandidate,
    fetched_path: Path,
    *,
    min_confidence: float = 0.4,
) -> CatalogEntry | None:
    """Build a temporary catalog entry from a fetched MusicXML/MXL (not MIDI)."""
    path = Path(fetched_path)
    if path.suffix.lower() in {".mid", ".midi"}:
        return None
    try:
        fp = parse_musicxml_fingerprint(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: discovery fingerprint failed: {exc}", file=sys.stderr)
        return None
    seq = [int(x) for x in fp.get("midi_sequence") or []]
    if len(seq) < 16:
        return None
    return CatalogEntry(
        id=f"discovered:{candidate.id}",
        title=candidate.title,
        edition=f"live discovery ({candidate.license})",
        edition_musicxml=path,
        composer=candidate.composer,
        license=candidate.license,
        imslp_work=candidate.imslp_work,
        key=candidate.key,
        min_confidence=min_confidence,
        match_notes="Fetched at identify-time (not a pre-bundled catalog copy).",
        typical_duration_sec=candidate.typical_duration_sec,
        midi_sequence=seq,
        ornaments=dict(fp.get("ornaments") or {}),
    )


def identify_with_discovery(
    hyp_midi: Sequence[int],
    *,
    catalog: list[CatalogEntry] | None = None,
    catalog_path: str | None = None,
    discovery_path: str | Path | None = None,
    discover: bool = True,
    duration_sec: float | None = None,
    inferred_key: str | None = None,
    cache_dir: Path | None = None,
    max_remote: int = 3,
) -> tuple[IdentificationResult, list[DiscoveryAttempt]]:
    """Local catalog match, then optional remote PD discovery on miss.

    Never pretends a pre-bundled catalog file is a live remote ID: remote hits
    use ``discovered:<id>`` piece ids and ``reason=discovery_match``.
    """
    attempts: list[DiscoveryAttempt] = []
    local = catalog if catalog is not None else load_catalog(catalog_path)
    result = identify_sequence(hyp_midi, catalog=local)
    if result.matched or not discover:
        return result, attempts

    seeds = load_discovery_index(discovery_path)
    if not seeds:
        attempts.append(
            DiscoveryAttempt(
                candidate_id="*",
                status="skipped",
                detail="empty_discovery_index",
            )
        )
        return result, attempts

    ranked = rank_discovery_candidates(
        seeds, duration_sec=duration_sec, inferred_key=inferred_key
    )
    cache = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="cnd-discover-"))
    remote_entries: list[CatalogEntry] = []

    local_ids = {e.id for e in local}
    for cand in ranked[: max(0, max_remote)]:
        # Skip remote seeds whose local twin is already bundled
        # (e.g. dallabaco_capriccio2_remote → dallabaco_capriccio2).
        base = cand.id.removesuffix("_remote")
        if cand.id in local_ids or base in local_ids:
            attempts.append(
                DiscoveryAttempt(
                    candidate_id=cand.id,
                    status="skipped",
                    detail=f"already_in_local_catalog ({base})",
                )
            )
            continue
        attempt = fetch_discovery_candidate(cand, cache_dir=cache)
        attempts.append(attempt)
        if attempt.status != "fetched" or not attempt.fetched_path:
            continue
        entry = entry_from_fetched_score(cand, Path(attempt.fetched_path))
        if entry is not None:
            remote_entries.append(entry)

    if not remote_entries:
        return result, attempts

    remote_result = identify_sequence(hyp_midi, catalog=remote_entries)
    if remote_result.matched:
        remote_result.reason = "discovery_match"
        remote_result.score_source = "catalog"
        return remote_result, attempts

    # Keep best remote metrics visible even on miss
    if remote_result.best is not None and (
        result.best is None or remote_result.confidence > result.confidence
    ):
        result = IdentificationResult(
            matched=False,
            confidence=remote_result.confidence,
            reason="discovery_low_confidence",
            entry=None,
            best=remote_result.best,
            candidates=(result.candidates or []) + remote_result.candidates,
            score_source="transcribe",
        )
    return result, attempts


def format_discovery_attempts(attempts: Sequence[DiscoveryAttempt]) -> str:
    if not attempts:
        return "(no discovery attempts)"
    lines = []
    for a in attempts:
        bit = f"  discover[{a.candidate_id}]: {a.status} — {a.detail}"
        if a.source_url:
            bit += f" [{a.source_url}]"
        lines.append(bit)
    return "\n".join(lines)
