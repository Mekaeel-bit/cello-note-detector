"""PDF → MusicXML via Audiveris (preferred) for IMSLP PDF-only works.

Design
------
Many public-domain cello scores on IMSLP are **PDF-only** (no MusicXML/MXL).
This module converts clean **engraved / typeset** PDFs to MusicXML with
Audiveris batch mode when the binary is available.

Quality limits (honest)
-----------------------
- Engraved modern typesets (Hamrefors, MuseScore exports, etc.) work best.
- Manuscripts / handwritten scans are **rejected by default** (poor OMR).
- Tenor/alto clef changes, grace notes, and dense 16ths can still mis-read;
  always spot-check MuseScore before trusting MATCH attach.
- Requires Audiveris ≥5.x on PATH or ``/opt/audiveris/bin/Audiveris``.
- Poppler ``pdftoppm`` is used only when Audiveris cannot ingest the PDF
  directly (rare); Audiveris 5.11 reads PDF natively.

Prefer per-movement PDFs (e.g. Caprice No.2) over complete-collection scans
so MATCH never ships an entire Caprices book for a single-clip take.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Preferred Audiveris locations (deb install + PATH).
_AUDIVERIS_CANDIDATES = (
    "Audiveris",
    "audiveris",
    "/opt/audiveris/bin/Audiveris",
    "/usr/bin/Audiveris",
)

# Filename / path hints that usually mean messy OMR inputs.
_MANUSCRIPT_HINTS = (
    "manuscript",
    "handwritten",
    "autograph",
    "holograph",
    "sketch",
    "ms.",
    "_ms_",
    "facsimile",
)
# Prefer engraved / modern typesets.
_ENGRAVED_HINTS = (
    "typeset",
    "engraved",
    "hamrefors",
    "musescore",
    "sibelius",
    "finale",
    "lilypond",
    "caprice_no",
    "capriccio",
)
# Whole-collection filenames — avoid when a single-movement PDF exists.
_COLLECTION_HINTS = (
    "complete",
    "11_capric",
    "11 capric",
    "all_capric",
    "caprices.pdf",
    "capricci.pdf",
    "1_through",
    "1-through",
    "collected",
)


@dataclass(frozen=True)
class PdfQuality:
    """Heuristic quality gate for OMR input selection."""

    path: Path
    score: float
    engraved_likely: bool
    manuscript_likely: bool
    collection_likely: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


@dataclass
class OmrResult:
    """Outcome of one PDF → MusicXML conversion."""

    status: str
    detail: str
    pdf_path: str | None = None
    musicxml_path: str | None = None
    mxl_path: str | None = None
    engine: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_audiveris(explicit: str | Path | None = None) -> Path | None:
    """Return path to Audiveris binary, or ``None`` if unavailable."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p
        return None
    env = os.environ.get("AUDIVERIS_BIN") or os.environ.get("CND_AUDIVERIS")
    if env:
        p = Path(env).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p
    which = shutil.which("Audiveris") or shutil.which("audiveris")
    if which:
        return Path(which)
    for cand in _AUDIVERIS_CANDIDATES:
        p = Path(cand)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def assess_pdf_quality(path: Path | str, *, title_hint: str | None = None) -> PdfQuality:
    """Score a PDF for OMR suitability (engraved preferred; manuscripts downranked)."""
    path = Path(path)
    name = path.name.lower()
    blob = f"{name} {(title_hint or '').lower()}"
    reasons: list[str] = []
    score = 0.5

    manuscript = any(h in blob for h in _MANUSCRIPT_HINTS)
    engraved = any(h in blob for h in _ENGRAVED_HINTS)
    collection = any(h in blob for h in _COLLECTION_HINTS)

    # Single-movement Caprice / numbered movement filenames score high.
    if re.search(r"(caprice|capriccio|ricercar|suite).{0,12}(no\.?\s*)?\d+", blob):
        score += 0.35
        reasons.append("numbered_movement_filename")
        engraved = True
    if engraved:
        score += 0.25
        reasons.append("engraved_or_typeset_hint")
    if manuscript:
        score -= 0.7
        reasons.append("manuscript_or_handwritten_hint")
    if collection:
        score -= 0.35
        reasons.append("complete_collection_filename")
    if path.suffix.lower() != ".pdf":
        score -= 0.5
        reasons.append("not_pdf")
    if path.is_file() and path.stat().st_size < 5_000:
        score -= 0.4
        reasons.append("tiny_file")

    return PdfQuality(
        path=path,
        score=float(max(-1.0, min(1.5, score))),
        engraved_likely=bool(engraved and not manuscript),
        manuscript_likely=bool(manuscript),
        collection_likely=bool(collection),
        reasons=tuple(reasons),
    )


def choose_best_pdf(
    candidates: Sequence[Path | str],
    *,
    title_hint: str | None = None,
    allow_manuscript: bool = False,
    min_score: float = 0.35,
) -> tuple[Path, PdfQuality] | None:
    """Pick the best engraved single-movement PDF from ``candidates``."""
    scored: list[PdfQuality] = []
    for raw in candidates:
        q = assess_pdf_quality(raw, title_hint=title_hint)
        if q.manuscript_likely and not allow_manuscript:
            continue
        if q.score < min_score:
            continue
        scored.append(q)
    if not scored:
        return None
    scored.sort(
        key=lambda q: (
            q.score,
            0 if q.collection_likely else 1,
            1 if q.engraved_likely else 0,
        ),
        reverse=True,
    )
    best = scored[0]
    return best.path, best


def pdf_to_musicxml(
    pdf_path: Path | str,
    output_dir: Path | str,
    *,
    audiveris_bin: str | Path | None = None,
    stem: str | None = None,
    allow_manuscript: bool = False,
    force: bool = False,
    timeout_sec: float = 600.0,
) -> OmrResult:
    """Run Audiveris batch ``-transcribe -export`` on a PDF → MusicXML/MXL.

    Returns an ``OmrResult``. On success, ``musicxml_path`` points at an
    unzipped ``.musicxml`` (extracted from Audiveris ``.mxl`` when needed).
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.is_file():
        return OmrResult(
            status="error",
            detail=f"PDF not found: {pdf_path}",
            pdf_path=str(pdf_path),
        )

    quality = assess_pdf_quality(pdf_path)
    if quality.manuscript_likely and not allow_manuscript:
        return OmrResult(
            status="rejected",
            detail=(
                "PDF looks like a manuscript/handwritten scan; "
                "prefer a clean engraved typeset (Hamrefors / MuseScore / etc.). "
                "Pass allow_manuscript=True to override."
            ),
            pdf_path=str(pdf_path),
            quality=quality.to_dict(),
        )

    bin_path = find_audiveris(audiveris_bin)
    if bin_path is None:
        return OmrResult(
            status="unavailable",
            detail=(
                "Audiveris not found. Install the Ubuntu package "
                "(https://github.com/Audiveris/audiveris/releases) or set "
                "AUDIVERIS_BIN. oemer is not wired as a default fallback "
                "(heavier deps / weaker on tenor-clef cello)."
            ),
            pdf_path=str(pdf_path),
            quality=quality.to_dict(),
        )

    out_stem = stem or pdf_path.stem
    # Avoid clobbering unrelated files; Audiveris names outputs after input stem.
    work = Path(tempfile.mkdtemp(prefix="cnd-omr-", dir=str(output_dir)))
    try:
        # Copy PDF into work dir so Audiveris output names stay predictable.
        local_pdf = work / f"{out_stem}.pdf"
        if pdf_path.resolve() != local_pdf.resolve():
            shutil.copy2(pdf_path, local_pdf)

        cmd = [
            str(bin_path),
            "-batch",
            "-transcribe",
            "-export",
            "-save",
            "-output",
            str(work),
        ]
        if force:
            cmd.append("-force")
        cmd.append(str(local_pdf))

        env = os.environ.copy()
        env.setdefault("JAVA_TOOL_OPTIONS", "-Djava.awt.headless=true")
        log_path = output_dir / f"{out_stem}_audiveris.log"
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return OmrResult(
                status="error",
                detail=f"Audiveris timed out after {timeout_sec:.0f}s",
                pdf_path=str(pdf_path),
                engine=str(bin_path),
                quality=quality.to_dict(),
            )

        log_path.write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""),
            encoding="utf-8",
        )

        mxl = work / f"{out_stem}.mxl"
        # Audiveris sometimes keeps the input basename.
        if not mxl.is_file():
            found = list(work.glob("*.mxl"))
            mxl = found[0] if found else mxl
        if not mxl.is_file():
            return OmrResult(
                status="error",
                detail=(
                    "Audiveris finished without an .mxl export "
                    f"(exit={proc.returncode}). See {log_path}"
                ),
                pdf_path=str(pdf_path),
                engine=str(bin_path),
                quality=quality.to_dict(),
                log_path=str(log_path),
            )

        # Promote artifacts next to caller output_dir.
        final_mxl = output_dir / f"{out_stem}.mxl"
        shutil.copy2(mxl, final_mxl)

        # Unzip to plain MusicXML for MuseScore / fingerprinting.
        from .musicxml_edition import resolve_edition_path

        xml_src = resolve_edition_path(final_mxl)
        final_xml = output_dir / f"{out_stem}.musicxml"
        shutil.copy2(xml_src, final_xml)

        return OmrResult(
            status="ok",
            detail="audiveris_export",
            pdf_path=str(pdf_path),
            musicxml_path=str(final_xml),
            mxl_path=str(final_mxl),
            engine=str(bin_path),
            quality=quality.to_dict(),
            log_path=str(log_path),
        )
    finally:
        # Keep work dir only on failure debugging — always clean temp copy.
        shutil.rmtree(work, ignore_errors=True)
