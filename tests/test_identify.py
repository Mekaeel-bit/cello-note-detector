"""Tests for local catalog song ID + published score attach / fallback."""

from __future__ import annotations

import json
from pathlib import Path

from cello_note_detector.identify.catalog import load_catalog
from cello_note_detector.identify.match import identify_sequence, score_against_entry
from cello_note_detector.identify.score import attach_catalog_score, reference_to_result

_REPO = Path(__file__).resolve().parents[1]
_CATALOG = _REPO / "data" / "catalog" / "catalog.json"
_REF = _REPO / "data" / "reference" / "bwv1007_prelude" / "prelude_monophonic_reference.json"


def test_load_catalog_includes_bwv1007():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    assert len(entries) >= 1
    by_id = {e.id: e for e in entries}
    assert "bwv1007_prelude" in by_id
    entry = by_id["bwv1007_prelude"]
    assert entry.num_notes == 606
    assert "Schirmer" in entry.edition or "Public Domain" in entry.license
    assert entry.reference_json.is_file()


def test_exact_reference_matches_with_high_confidence():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    entry = next(e for e in entries if e.id == "bwv1007_prelude")
    # Exact prefix of the written Prelude.
    hyp = entry.midi_sequence[:64]
    score = score_against_entry(hyp, entry)
    assert score.window_accuracy == 1.0
    assert score.motif_accuracy == 1.0
    assert score.confidence >= 0.9
    assert score.above_threshold

    result = identify_sequence(hyp, catalog=entries)
    assert result.matched
    assert result.entry is not None
    assert result.entry.id == "bwv1007_prelude"
    assert result.score_source == "catalog"


def test_noisy_sequence_still_matches_opening():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    entry = next(e for e in entries if e.id == "bwv1007_prelude")
    # Keep bars 1–2 intact; insert extras later (performance over-segmentation).
    hyp = list(entry.midi_sequence[:64])
    hyp.insert(24, hyp[23])
    hyp.insert(40, 60)
    result = identify_sequence(hyp, catalog=entries)
    assert result.matched
    assert result.confidence >= entry.min_confidence


def test_unrelated_sequence_falls_back():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    # Ascending/descending C-major — not the Prelude (long enough to score).
    hyp = [48, 50, 52, 53, 55, 57, 59, 60] * 4
    result = identify_sequence(hyp, catalog=entries)
    assert not result.matched
    assert result.score_source == "transcribe"
    assert result.reason in {"low_confidence", "too_few_notes"}


def test_short_clip_too_few_notes_falls_back():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    result = identify_sequence([48, 50, 52, 53, 55, 57, 59, 60], catalog=entries)
    assert not result.matched
    assert result.reason == "too_few_notes"


def test_empty_catalog_falls_back():
    result = identify_sequence([43, 50, 59, 57, 59, 50, 59, 50] * 2, catalog=[])
    assert not result.matched
    assert result.reason == "empty_catalog"


def test_reference_to_result_builds_exportable_timeline():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    entry = next(e for e in entries if e.id == "bwv1007_prelude")
    built = reference_to_result(entry, audio_path="demo.wav", bpm=60.0)
    # Playable catalog edition (re-parsed LilyPond), not the eval reference tail.
    assert built["num_notes"] >= 600
    assert built["events"][0]["midi"] == 43
    assert built["events"][0]["duration_value"] == "sixteenth"
    assert built["key"]["name"] == "G major"
    assert built["score_source"] == "catalog"
    midis = [e["midi"] for e in built["events"]]
    assert min(midis) >= 36  # cello C2 floor — no ledger-line garbage
    assert max(midis) <= 72


def test_catalog_score_json_playable_and_matches_verified_prefix():
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    entry = next(e for e in entries if e.id == "bwv1007_prelude")
    assert entry.score_json is not None and entry.score_json.is_file()
    score = entry.score_data
    midis = [int(m) for m in score["midi_sequence"]]
    assert min(midis) >= 36
    assert midis[:128] == entry.midi_sequence[:128]


def test_attach_catalog_score_writes_artifacts(tmp_path: Path):
    entries = load_catalog(_CATALOG, repo_root=_REPO)
    entry = next(e for e in entries if e.id == "bwv1007_prelude")
    hyp = entry.midi_sequence[:32]
    identification = identify_sequence(hyp, catalog=entries)
    assert identification.matched

    result = {
        "audio_path": "data/multi_note_phrases/Bach-cello-suite-1.wav",
        "tempo": {"bpm": 67.0},
        "notes": [{"midi": m} for m in hyp],
        "events": [],
    }
    out = attach_catalog_score(
        result,
        identification,
        output_dir=tmp_path,
        stem="Bach-cello-suite-1",
        export_musicxml=True,
        export_midi=True,
    )
    assert out["score_source"] == "catalog"
    assert out["identification"]["matched"] is True
    assert out["identification"]["score_kind"] == "catalog_edition"
    xml_path = tmp_path / "Bach-cello-suite-1.musicxml"
    assert xml_path.is_file()
    assert (tmp_path / "Bach-cello-suite-1.mid").is_file()
    assert (tmp_path / "Bach-cello-suite-1_identification.json").is_file()
    assert (tmp_path / "Bach-cello-suite-1_catalog_reference.json").is_file()
    assert (tmp_path / "Bach-cello-suite-1_catalog_score.json").is_file()
    meta = json.loads(
        (tmp_path / "Bach-cello-suite-1_identification.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["title"].startswith("Bach Cello Suite")
    assert "Schirmer" in meta["edition"] or "Mutopia" in meta["edition"]
    xml = xml_path.read_text(encoding="utf-8")
    assert "Bach Cello Suite" in xml
    # No sub-C2 garbage in the primary catalog MusicXML.
    import re

    octaves = [int(o) for o in re.findall(r"<octave>(\d)</octave>", xml)]
    assert octaves
    assert min(octaves) >= 2


def test_attach_no_match_keeps_transcription_fallback(tmp_path: Path):
    identification = identify_sequence(
        [48, 50, 52, 53, 55, 57, 59, 60] * 2,
        catalog=load_catalog(_CATALOG, repo_root=_REPO),
    )
    assert not identification.matched
    result = {
        "audio_path": "x.wav",
        "notes": [],
        "events": [],
    }
    out = attach_catalog_score(
        result,
        identification,
        output_dir=tmp_path,
        stem="unknown",
        export_musicxml=False,
        export_midi=False,
    )
    assert out["identification"]["fallback"] == "audio_transcription"
    assert out["score_source"] == "transcribe"
    assert not (tmp_path / "unknown.musicxml").exists()


def test_reference_json_schema_stable():
    data = json.loads(_REF.read_text(encoding="utf-8"))
    assert len(data["midi_sequence"]) == data["num_notes"] == 606
    assert len(data["events"]) == 606
