"""PDF quality gates + MusicXML clip excerpt helpers (OMR path)."""

from __future__ import annotations

from pathlib import Path

from cello_note_detector.identify.musicxml_edition import (
    excerpt_musicxml_by_notes,
    parse_musicxml_fingerprint,
    resolve_clip_note_span,
)
from cello_note_detector.identify.omr import (
    assess_pdf_quality,
    choose_best_pdf,
    find_audiveris,
)

_REPO = Path(__file__).resolve().parents[1]
_CAPRICE2_PDF = (
    _REPO
    / "data"
    / "catalog"
    / "pieces"
    / "dallabaco_capriccio2"
    / "source"
    / "omr"
    / "IMSLP618567-Hamrefors-Caprice_No_2.pdf"
)
_CAPRICE2_XML = (
    _REPO / "data" / "catalog" / "pieces" / "dallabaco_capriccio2" / "edition.musicxml"
)


def test_prefer_engraved_single_caprice_over_complete_book():
    names = [
        "PMLP626508-Dall'Abaco_Complete_Caprices.pdf",
        "PMLP626508-Dall'Abaco_Caprice_No._2.pdf",
        "PMLP626508-manuscript_autograph.pdf",
    ]
    best = choose_best_pdf(names, title_hint="Capriccio No.2")
    assert best is not None
    assert "Caprice_No._2" in best[0].name
    assert best[1].engraved_likely
    assert not best[1].manuscript_likely


def test_manuscript_pdf_rejected_by_default():
    q = assess_pdf_quality("something_handwritten_manuscript.pdf")
    assert q.manuscript_likely
    assert q.score < 0.35


def test_hamrefors_caprice2_pdf_bundled_and_scored():
    assert _CAPRICE2_PDF.is_file()
    q = assess_pdf_quality(_CAPRICE2_PDF)
    assert q.score >= 0.5
    assert not q.manuscript_likely
    assert not q.collection_likely


def test_resolve_clip_keeps_full_movement_when_duration_matches():
    span = resolve_clip_note_span(
        ref_notes=480,
        hyp_notes=200,
        window_offset=0,
        window_mode="hyp_in_ref",
        audio_duration_sec=176.0,
        typical_duration_sec=176.0,
    )
    assert span is None  # whole Caprice 2


def test_resolve_clip_trims_short_excerpt():
    span = resolve_clip_note_span(
        ref_notes=480,
        hyp_notes=64,
        window_offset=40,
        window_mode="hyp_in_ref",
        audio_duration_sec=40.0,
        typical_duration_sec=176.0,
    )
    assert span == (40, 104)


def test_excerpt_musicxml_by_notes_shrinks_score(tmp_path: Path):
    assert _CAPRICE2_XML.is_file()
    full = parse_musicxml_fingerprint(_CAPRICE2_XML)
    dest = tmp_path / "excerpt.musicxml"
    excerpt_musicxml_by_notes(_CAPRICE2_XML, dest, start_note=0, end_note=80)
    part = parse_musicxml_fingerprint(dest)
    # Measure-granularity keep can include a few notes past end_note.
    assert part["num_notes"] < full["num_notes"]
    assert part["num_notes"] <= 120
    assert part["num_measures"] < full["num_measures"]


def test_find_audiveris_optional():
    # Soft check — CI may or may not have Audiveris; API must not raise.
    path = find_audiveris()
    assert path is None or path.is_file()
