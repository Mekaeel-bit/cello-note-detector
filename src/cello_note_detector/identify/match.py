"""Match a transcribed MIDI sequence against the local catalog."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from .catalog import CatalogEntry, load_catalog

# Distinctive opening length for motif checks (Prelude bars 1–2 ≈ 16 sixteenths).
_MOTIF_LEN = 16


@dataclass
class MatchScore:
    """Similarity metrics for one catalog entry."""

    entry_id: str
    title: str
    edition: str
    confidence: float
    window_accuracy: float
    window_matched: int
    window_compared: int
    window_offset: int
    window_mode: str
    run_ratio: float
    longest_run: int
    hyp_notes: int
    ref_notes: int
    motif_accuracy: float = 0.0
    octave_shifted: int = 0
    above_threshold: bool = False
    min_confidence: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IdentificationResult:
    """Best catalog match (or no-match) for a pitch sequence."""

    matched: bool
    confidence: float
    reason: str
    entry: CatalogEntry | None = None
    best: MatchScore | None = None
    candidates: list[MatchScore] = field(default_factory=list)
    score_source: str = "transcribe"

    def to_metadata(self) -> dict[str, Any]:
        """JSON-serializable identification block for the notes export."""
        meta: dict[str, Any] = {
            "matched": self.matched,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "score_source": self.score_source,
            "catalog_size": len(self.candidates),
        }
        if self.entry is not None:
            meta["piece_id"] = self.entry.id
            meta["title"] = self.entry.title
            meta["composer"] = self.entry.composer
            meta["edition"] = self.entry.edition
            meta["license"] = self.entry.license
            meta["imslp_work"] = self.entry.imslp_work
            meta["key"] = self.entry.key
            meta["reference_json"] = str(self.entry.reference_json)
            if self.entry.midi_path is not None:
                meta["reference_midi"] = str(self.entry.midi_path)
        if self.best is not None:
            meta["metrics"] = {
                "window_accuracy": round(self.best.window_accuracy, 4),
                "window_matched": self.best.window_matched,
                "window_compared": self.best.window_compared,
                "window_offset": self.best.window_offset,
                "window_mode": self.best.window_mode,
                "motif_accuracy": round(self.best.motif_accuracy, 4),
                "run_ratio": round(self.best.run_ratio, 4),
                "longest_run": self.best.longest_run,
                "octave_shifted": self.best.octave_shifted,
                "hyp_notes": self.best.hyp_notes,
                "ref_notes": self.best.ref_notes,
                "min_confidence": self.best.min_confidence,
            }
        if self.candidates:
            meta["candidates"] = [
                {
                    "piece_id": c.entry_id,
                    "title": c.title,
                    "confidence": round(c.confidence, 4),
                    "above_threshold": c.above_threshold,
                }
                for c in self.candidates[:5]
            ]
        return meta


def _longest_common_run(a: Sequence[int], b: Sequence[int]) -> int:
    """Longest contiguous common substring length (exact pitch run)."""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            if int(ca) == int(cb):
                val = prev[j] + 1
                cur.append(val)
                if val > best:
                    best = val
            else:
                cur.append(0)
        prev = cur
    return best


def _best_window(hyp: Sequence[int], ref: Sequence[int]) -> dict[str, Any]:
    """Slide the shorter sequence over the longer; report best exact-match rate."""
    if not hyp or not ref:
        return {
            "accuracy": 0.0,
            "matched": 0,
            "compared": 0,
            "offset": 0,
            "mode": "empty",
        }

    best = {
        "accuracy": -1.0,
        "matched": 0,
        "compared": 0,
        "offset": 0,
        "mode": "",
    }

    def _consider(
        window_a: Sequence[int],
        window_b: Sequence[int],
        offset: int,
        mode: str,
    ) -> None:
        matched = sum(1 for x, y in zip(window_a, window_b) if x == y)
        compared = len(window_a)
        acc = matched / compared if compared else 0.0
        if acc > best["accuracy"]:
            best.update(
                accuracy=acc,
                matched=matched,
                compared=compared,
                offset=offset,
                mode=mode,
            )

    if len(hyp) <= len(ref):
        for off in range(0, len(ref) - len(hyp) + 1):
            _consider(hyp, ref[off : off + len(hyp)], off, "hyp_in_ref")
    if len(ref) <= len(hyp):
        for off in range(0, len(hyp) - len(ref) + 1):
            _consider(hyp[off : off + len(ref)], ref, off, "ref_in_hyp")
    return best


def _shift_octave(seq: Sequence[int], octaves: int) -> list[int]:
    return [int(m) + 12 * octaves for m in seq]


def _passes_gates(
    *,
    confidence: float,
    window_accuracy: float,
    motif_accuracy: float,
    longest_run: int,
    hyp_len: int,
    min_confidence: float,
) -> bool:
    """Accept only strong contiguous evidence (avoid subsequence false matches)."""
    if confidence < min_confidence:
        return False

    # Real take of a known solo: opening motif intact + a solid contiguous run.
    if (
        hyp_len >= 32
        and motif_accuracy >= 0.75
        and longest_run >= 12
    ):
        return True

    # Clean excerpt of the written piece.
    min_acc = 0.75 if hyp_len < 24 else 0.55
    if window_accuracy >= min_acc and longest_run >= min(16, max(8, hyp_len // 2)):
        return True

    return False


def score_against_entry(
    hyp_midi: Sequence[int],
    entry: CatalogEntry,
    *,
    try_octave_shifts: tuple[int, ...] = (0, -1, 1),
) -> MatchScore:
    """Score ``hyp_midi`` against one catalog entry.

    Uses contiguous evidence only:
    - best full-sequence window accuracy (clean excerpts)
    - opening-motif accuracy (ref head located inside the hypothesis)
    - longest exact pitch run / run_ratio (shared contiguous fragment)

    Global subsequence LCS is intentionally avoided — diatonic fragments of
    unrelated solos otherwise false-match long catalog pieces.
    """
    hyp = [int(m) for m in hyp_midi]
    ref = entry.midi_sequence
    best_pack: dict[str, Any] | None = None
    best_shift = 0
    best_run = 0
    best_confidence = -1.0
    best_run_ratio = 0.0
    best_motif = 0.0

    motif = ref[: min(_MOTIF_LEN, len(ref))]

    for shift in try_octave_shifts:
        cand = _shift_octave(hyp, shift) if shift else hyp
        window = _best_window(cand, ref)

        if motif and len(cand) >= len(motif):
            motif_acc = float(_best_window(motif, cand)["accuracy"])
        elif motif and cand:
            motif_acc = float(_best_window(cand, motif)["accuracy"])
        else:
            motif_acc = 0.0

        # Cap run search on very long pairs for latency.
        max_n = 900
        run = _longest_common_run(cand[:max_n], ref[:max_n])
        run_denom = max(min(len(cand), len(ref), 64), 1)
        run_ratio = min(run, run_denom) / run_denom

        excerpt_conf = 0.7 * float(window["accuracy"]) + 0.3 * run_ratio
        performance_conf = 0.65 * motif_acc + 0.35 * run_ratio
        confidence = max(excerpt_conf, performance_conf)

        if confidence > best_confidence:
            best_confidence = confidence
            best_pack = window
            best_shift = shift
            best_run = run
            best_run_ratio = run_ratio
            best_motif = motif_acc

    assert best_pack is not None
    above = _passes_gates(
        confidence=float(best_confidence),
        window_accuracy=float(best_pack["accuracy"]),
        motif_accuracy=float(best_motif),
        longest_run=int(best_run),
        hyp_len=len(hyp),
        min_confidence=entry.min_confidence,
    )
    return MatchScore(
        entry_id=entry.id,
        title=entry.title,
        edition=entry.edition,
        confidence=float(best_confidence),
        window_accuracy=float(best_pack["accuracy"]),
        window_matched=int(best_pack["matched"]),
        window_compared=int(best_pack["compared"]),
        window_offset=int(best_pack["offset"]),
        window_mode=str(best_pack["mode"]),
        run_ratio=float(best_run_ratio),
        longest_run=int(best_run),
        hyp_notes=len(hyp),
        ref_notes=len(ref),
        motif_accuracy=float(best_motif),
        octave_shifted=int(best_shift),
        above_threshold=above,
        min_confidence=entry.min_confidence,
    )


def identify_sequence(
    hyp_midi: Sequence[int],
    *,
    catalog: list[CatalogEntry] | None = None,
    catalog_path: str | None = None,
    min_notes: int = 16,
) -> IdentificationResult:
    """Identify ``hyp_midi`` against the local catalog.

    Returns a no-match result (never raises) when the catalog is empty, the
    hypothesis is too short, or confidence is below every entry threshold.
    """
    entries = catalog if catalog is not None else load_catalog(catalog_path)
    hyp = [int(m) for m in hyp_midi]

    if not entries:
        return IdentificationResult(
            matched=False,
            confidence=0.0,
            reason="empty_catalog",
            score_source="transcribe",
        )
    if len(hyp) < min_notes:
        return IdentificationResult(
            matched=False,
            confidence=0.0,
            reason="too_few_notes",
            candidates=[],
            score_source="transcribe",
        )

    scored = [score_against_entry(hyp, entry) for entry in entries]
    scored.sort(key=lambda s: s.confidence, reverse=True)
    best = scored[0]
    entry_map = {e.id: e for e in entries}
    winner = entry_map[best.entry_id]

    if best.above_threshold:
        return IdentificationResult(
            matched=True,
            confidence=best.confidence,
            reason="catalog_match",
            entry=winner,
            best=best,
            candidates=scored,
            score_source="catalog",
        )

    return IdentificationResult(
        matched=False,
        confidence=best.confidence,
        reason="low_confidence",
        entry=None,
        best=best,
        candidates=scored,
        score_source="transcribe",
    )
