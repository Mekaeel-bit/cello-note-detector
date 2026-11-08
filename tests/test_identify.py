"""Tests for extensible PD MusicXML catalog song ID + verbatim score attach."""

from __future__ import annotations

import json
import re
from pathlib import Path

from cello_note_detector.identify.catalog import (
    format_catalog_listing,
    load_catalog,
    search_catalog,
)
from cello_note_detector.identify.match import identify_sequence, score_against_entry
from cello_note_detector.identify.musicxml_edition import (
    copy_edition_musicxml,
    parse_musicxml_fingerprint,
)
from cello_note_detector.identify.score import attach_catalog_score

_REPO = Path(__file__).resolve().parents[1]
_CATALOG = _REPO / "data" / "catalog" / "catalog.json"
_BACH_XML = (
    _REPO / "data" / "catalog" / "pieces" / "bwv1007_prelude" / "edition.musicxml"
)
_GAB_XML = (
    _REPO / "data" / "catalog" / "pieces" / "gabrielli_ricercari" / "edition.musicxml"
)


def test_load_catalog_has_bach_and_gabrielli():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    ids = {e.id for e in entries}
    assert "bwv1007_prelude" in ids
    assert "gabrielli_ricercari" in ids
    bach = next(e for e in entries if e.id == "bwv1007_prelude")
    assert bach.edition_musicxml.is_file()
    assert bach.num_notes >= 500
    assert bach.ornaments.get("has_slurs") is True


def test_search_catalog_pd_lookup():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    assert {e.id for e in search_catalog("", entries=entries)} == {
        e.id for e in entries
    }
    bach_hits = search_catalog("bach", entries=entries)
    assert [e.id for e in bach_hits] == ["bwv1007_prelude"]
    gab_hits = search_catalog("Gabrielli", entries=entries)
    assert [e.id for e in gab_hits] == ["gabrielli_ricercari"]
    assert "gabrielli_ricercari" in format_catalog_listing(gab_hits)
    assert search_catalog("not-a-real-piece-xyz", entries=entries) == []


def test_bach_edition_musicxml_playable_octaves_and_slurs():
    assert _BACH_XML.is_file()
    text = _BACH_XML.read_text(encoding="utf-8")
    assert "slur" in text
    octaves = [int(o) for o in re.findall(r"<octave>(\d)</octave>", text)]
    assert octaves
    assert min(octaves) >= 2
    assert max(octaves) <= 5
    fp = parse_musicxml_fingerprint(_BACH_XML)
    assert fp["num_notes"] >= 500
    assert fp["ornaments"]["has_slurs"] is True


def test_exact_fingerprint_matches_bach():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    entry = next(e for e in entries if e.id == "bwv1007_prelude")
    hyp = entry.midi_sequence[:64]
    score = score_against_entry(hyp, entry)
    assert score.window_accuracy == 1.0
    assert score.above_threshold
    result = identify_sequence(hyp, catalog=entries)
    assert result.matched
    assert result.entry is not None
    assert result.entry.id == "bwv1007_prelude"


def test_unrelated_sequence_falls_back():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    hyp = [48, 50, 52, 53, 55, 57, 59, 60] * 4
    result = identify_sequence(hyp, catalog=entries)
    assert not result.matched
    assert result.score_source == "transcribe"


def test_empty_catalog_falls_back():
    result = identify_sequence([43, 50, 59, 57, 59, 50, 59, 50] * 2, catalog=[])
    assert not result.matched
    assert result.reason == "empty_catalog"


def test_attach_copies_edition_musicxml_verbatim(tmp_path: Path):
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    entry = next(e for e in entries if e.id == "bwv1007_prelude")
    identification = identify_sequence(entry.midi_sequence[:48], catalog=entries)
    assert identification.matched

    result = {
        "audio_path": "data/multi_note_phrases/Bach-cello-suite-1.wav",
        "duration_sec": 159.0,
        "tempo": {"bpm": 67.0},
        "notes": [{"midi": m} for m in entry.midi_sequence[:48]],
        "events": [],
    }
    out = attach_catalog_score(
        result,
        identification,
        output_dir=tmp_path,
        stem="Bach-cello-suite-1",
        export_musicxml=True,
        export_midi=False,
        catalog_entries=entries,
    )
    xml_path = tmp_path / "Bach-cello-suite-1.musicxml"
    assert xml_path.is_file()
    assert out["identification"]["score_kind"] == "verbatim_pd_edition"
    # Byte-identical to the catalog edition (verbatim copy).
    assert xml_path.read_bytes() == entry.edition_musicxml.read_bytes()
    text = xml_path.read_text(encoding="utf-8")
    assert "slur" in text
    octaves = [int(o) for o in re.findall(r"<octave>(\d)</octave>", text)]
    assert min(octaves) >= 2


def test_copy_edition_helper(tmp_path: Path):
    dest = tmp_path / "out.musicxml"
    written = copy_edition_musicxml(_BACH_XML, dest)
    assert written == dest
    assert dest.read_bytes() == _BACH_XML.read_bytes()


def test_gabrielli_edition_present_and_fingerprintable():
    assert _GAB_XML.is_file()
    text = _GAB_XML.read_text(encoding="utf-8")
    assert "1st Ricercar" in text
    assert "3</beats>" in text or ">3</beats>" in text
    fp = parse_musicxml_fingerprint(_GAB_XML)
    assert fp["num_notes"] >= 200
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    gab = next(e for e in entries if e.id == "gabrielli_ricercari")
    # Opening G2–A2–Bb2 of Ricercar 1 (sounding / Gomez-aligned fingerprint).
    assert gab.midi_sequence[:3] == [43, 45, 46]
    assert gab.num_notes >= 200


def test_attach_no_match_keeps_transcription_fallback(tmp_path: Path):
    identification = identify_sequence(
        [48, 50, 52, 53, 55, 57, 59, 60] * 2,
        catalog=load_catalog(_CATALOG, repo_root=_REPO),
    )
    assert not identification.matched
    out = attach_catalog_score(
        {"audio_path": "x.wav", "notes": [], "events": []},
        identification,
        output_dir=tmp_path,
        stem="unknown",
        export_musicxml=False,
        export_midi=False,
    )
    assert out["identification"]["fallback"] == "audio_transcription"
    assert out["score_source"] == "transcribe"
