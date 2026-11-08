"""CLI: pyin note events + rhythm quantization + optional MERT embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cello_note_detector.audio.load import TARGET_SR, load_mono
from cello_note_detector.audio.mert import (
    DEFAULT_MODEL_ID,
    extract_mert_embeddings,
    load_mert,
    summarize_embeddings,
)
from cello_note_detector.audio.pitch_baseline import (
    DEFAULT_GAP_TOLERANCE,
    DEFAULT_HOP_LENGTH,
    DEFAULT_MEDIAN_FILTER_SIZE,
    DEFAULT_MIN_DURATION,
    transcribe_notes,
)
from cello_note_detector.decode.events import note_events_to_dicts
from cello_note_detector.decode.key import infer_key, parse_key
from cello_note_detector.decode.rhythm import (
    DEFAULT_GRID_BEATS,
    DEFAULT_MIN_REST_SEC,
    DEFAULT_QUANTIZE,
    analyze_rhythm,
    resolve_quantize_preset,
)
from cello_note_detector.export import write_midi, write_musicxml
from cello_note_detector.identify import (
    attach_catalog_score,
    format_catalog_listing,
    format_discovery_attempts,
    identify_with_discovery,
    load_catalog,
    load_discovery_index,
    search_catalog,
)
from cello_note_detector.identify.catalog import DEFAULT_CATALOG_PATH
from cello_note_detector.identify.discover import DEFAULT_DISCOVERY_PATH

# Repo root: .../src/cello_note_detector/cli.py → parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDINGS = _REPO_ROOT / "data" / "Recordings"
DEFAULT_PHRASES = _REPO_ROOT / "data" / "multi_note_phrases"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs"
SCORE_SOURCES = ("auto", "transcribe", "catalog")


def _discover_wavs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".wav":
            raise SystemExit(f"Expected a .wav file, got: {path}")
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Path not found: {path}")
    wavs = sorted(path.glob("*.wav")) + sorted(path.glob("*.WAV"))
    # de-dupe while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for w in wavs:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    if not unique:
        raise SystemExit(f"No WAV files found under: {path}")
    return unique


def _merge_note_rhythm(note_dicts: list[dict], note_rhythm: list[dict]) -> list[dict]:
    """Attach quantized rhythm fields onto each note dict."""
    out: list[dict] = []
    for n, r in zip(note_dicts, note_rhythm):
        merged = dict(n)
        merged.update(r)
        out.append(merged)
    return out


def _resolve_score_source(
    *,
    score_source: str | None,
    identify: bool,
) -> str:
    """Default score source: ``auto`` when identifying, else ``transcribe``."""
    if score_source:
        return score_source
    return "auto" if identify else "transcribe"


def _process_one(
    wav_path: Path,
    *,
    output_dir: Path,
    mert_bundle,
    save_embeddings: bool,
    skip_mert: bool,
    skip_rhythm: bool,
    min_duration: float,
    gap_tolerance: float,
    median_filter_size: int,
    use_onsets: bool,
    quantize_cfg: dict,
    bpm_override: float | None,
    key_override: str | None,
    no_edge_rests: bool,
    export_musicxml: bool,
    export_midi: bool,
    identify: bool = False,
    score_source: str = "transcribe",
    catalog_entries: list | None = None,
    discover: bool = True,
    discovery_path: str | None = None,
) -> dict:
    audio, sr = load_mono(wav_path, sr=TARGET_SR)
    duration_s = float(len(audio) / sr)

    events, seg_meta = transcribe_notes(
        audio,
        sr=sr,
        min_duration=min_duration,
        gap_tolerance=gap_tolerance,
        median_filter_size=median_filter_size,
        use_onsets=use_onsets,
        hop_length=DEFAULT_HOP_LENGTH,
    )
    note_dicts = note_events_to_dicts(events)

    # Key from detected pitches (or CLI). Never from the filename.
    if key_override:
        try:
            key_info = parse_key(key_override)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        key_info = infer_key(events)

    rhythm_info: dict | None = None
    rests: list[dict] = []
    timeline: list[dict] = []
    if not skip_rhythm:
        rhythm_info = analyze_rhythm(
            audio,
            events,
            sr=sr,
            grid_beats=float(quantize_cfg["grid_beats"]),
            min_rest_sec=float(quantize_cfg["min_rest_sec"]),
            min_rest_beats=float(quantize_cfg["min_rest_beats"]),
            min_note_beats=float(quantize_cfg["min_note_beats"]),
            catalog=quantize_cfg["catalog"],
            include_leading_rests=not no_edge_rests,
            include_trailing_rests=not no_edge_rests,
            bpm_override=bpm_override,
            score_intent=bool(quantize_cfg.get("score_intent", False)),
        )
        note_dicts = _merge_note_rhythm(note_dicts, rhythm_info["note_rhythm"])
        rests = rhythm_info["rests"]
        timeline = rhythm_info["events"]

    mert_info: dict | None = None
    emb_path: str | None = None
    if not skip_mert:
        embeddings = extract_mert_embeddings(audio, mert_bundle, sr=sr)
        mert_info = {
            "model_id": mert_bundle.model_id,
            "device": str(mert_bundle.device),
            "summary": summarize_embeddings(embeddings),
        }
        if save_embeddings:
            emb_file = output_dir / f"{wav_path.stem}_mert_embeddings.npy"
            np.save(emb_file, embeddings)
            emb_path = str(emb_file)
            mert_info["embeddings_path"] = emb_path

    result: dict = {
        "audio_path": str(wav_path),
        "sample_rate": sr,
        "duration_sec": round(duration_s, 4),
        "pitch_method": "librosa.pyin",
        "cello_range": {"fmin": "C2", "fmax": "A5"},
        "segmentation": seg_meta,
        "key": key_info,
        "num_notes": len(note_dicts),
        "notes": note_dicts,
        "mert": mert_info,
    }

    if rhythm_info is not None:
        result["tempo"] = rhythm_info["tempo"]
        result["rhythm"] = {
            "quantize": quantize_cfg["name"],
            "grid_beats": rhythm_info["grid_beats"],
            "min_rest_sec": rhythm_info["min_rest_sec"],
            "min_rest_beats": rhythm_info["min_rest_beats"],
            "min_note_beats": rhythm_info["min_note_beats"],
            "score_intent": rhythm_info.get("score_intent", False),
            "equal_stream": rhythm_info.get("equal_stream"),
            "method": (
                "score-intent equal-stream reconstruction"
                if rhythm_info.get("equal_stream")
                else (
                    "spectral-onset + beat_track blend; "
                    "coarser duration catalog snap"
                )
            ),
        }
        result["num_rests"] = rhythm_info["num_rests"]
        result["rests"] = rests
        result["events"] = timeline

    # Song ID: local bundled catalog first, then optional remote PD discovery.
    # Bundled files under data/catalog/pieces/ are NOT a live ID by themselves —
    # MATCH only when identify copies an edition into output_dir.
    # Unknown / low confidence → audio transcription (never fails the run).
    want_identify = identify or score_source in ("auto", "catalog")
    use_catalog_score = False
    if want_identify:
        hyp_midi = [int(n["midi"]) for n in note_dicts]
        key_name = None
        if isinstance(key_info, dict):
            key_name = key_info.get("name")
        identification, discovery_attempts = identify_with_discovery(
            hyp_midi,
            catalog=catalog_entries if catalog_entries is not None else [],
            discover=discover and score_source != "transcribe",
            duration_sec=duration_s,
            inferred_key=key_name,
            discovery_path=discovery_path,
            cache_dir=output_dir / "_discovery_cache",
        )
        if score_source == "transcribe":
            # Still record match metadata, but keep transcription sheet output.
            identification.score_source = "transcribe"
            if identification.matched:
                identification.reason = "catalog_match_metadata_only"
        elif score_source == "catalog" and not identification.matched:
            identification.score_source = "transcribe"
            identification.reason = (
                identification.reason + "_fallback_transcribe"
                if identification.reason
                else "catalog_miss_fallback_transcribe"
            )
        elif score_source == "auto" and not identification.matched:
            identification.score_source = "transcribe"

        use_catalog_score = (
            identification.matched
            and identification.score_source == "catalog"
            and score_source in ("auto", "catalog")
        )
        result = attach_catalog_score(
            result,
            identification,
            output_dir=output_dir,
            stem=wav_path.stem,
            # Defer MusicXML/MIDI to the block below when using transcription;
            # when using catalog, attach_catalog_score copies the PD edition.
            export_musicxml=export_musicxml and use_catalog_score,
            export_midi=export_midi and use_catalog_score,
            bpm=bpm_override,
            catalog_entries=catalog_entries,
        )
        if discovery_attempts:
            result["identification"]["discovery"] = [
                a.to_dict() for a in discovery_attempts
            ]
            result["_discovery_log"] = format_discovery_attempts(discovery_attempts)
        result["score_source"] = (
            "catalog" if use_catalog_score else "transcribe"
        )
    else:
        result["score_source"] = "transcribe"

    json_path = output_dir / f"{wav_path.stem}_notes.json"

    if export_musicxml:
        if use_catalog_score:
            # Primary .musicxml already written from the PD catalog edition.
            # Also keep the performance dump for A/B comparison.
            if not skip_rhythm and result.get("events"):
                tx_path = output_dir / f"{wav_path.stem}_transcribed.musicxml"
                write_musicxml(result, tx_path, title=f"{wav_path.stem} (transcribed)")
                result["_transcribed_musicxml_path"] = str(tx_path)
                ident = result.get("identification")
                if isinstance(ident, dict):
                    ident["transcribed_musicxml"] = str(tx_path)
        elif skip_rhythm or not result.get("events"):
            print(
                f"  warning: MusicXML skipped for {wav_path.name} "
                "(needs rhythm/events; omit --skip-rhythm)"
            )
        else:
            xml_path = output_dir / f"{wav_path.stem}.musicxml"
            write_musicxml(result, xml_path, title=wav_path.stem)
            result["_musicxml_path"] = str(xml_path)

    if export_midi and not use_catalog_score:
        if skip_rhythm or not result.get("events"):
            print(
                f"  warning: MIDI skipped for {wav_path.name} "
                "(needs rhythm/events; omit --skip-rhythm)"
            )
        else:
            mid_path = output_dir / f"{wav_path.stem}.mid"
            write_midi(result, mid_path)
            result["_midi_path"] = str(mid_path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["_json_path"] = str(json_path)

    return result


def _print_summary(result: dict) -> None:
    print(f"\n=== {Path(result['audio_path']).name} ===")
    tempo = result.get("tempo")
    tempo_bit = f"  tempo={tempo['bpm']:.1f} BPM" if tempo else ""
    key = result.get("key")
    key_bit = f"  key={key['name']}" if key else ""
    rests_bit = (
        f"  rests={result.get('num_rests', 0)}"
        if "num_rests" in result
        else ""
    )
    print(
        f"duration={result['duration_sec']:.2f}s  "
        f"notes={result['num_notes']}"
        f"{rests_bit}"
        f"{tempo_bit}"
        f"{key_bit}  "
        f"onsets={result['segmentation'].get('num_onsets', 0)}  "
        f"json={result['_json_path']}"
    )
    preview = result["notes"][:12]
    for n in preview:
        dur_val = n.get("duration_value")
        dur_bit = f"  {dur_val}" if dur_val else ""
        print(
            f"  note  {n['onset']:6.2f}–{n['offset']:6.2f}s  "
            f"{n['note']:>4}  midi={n['midi']:3d}  "
            f"conf={n['confidence']:.2f}{dur_bit}"
        )
    if result["num_notes"] > len(preview):
        print(f"  … {result['num_notes'] - len(preview)} more notes")

    if result.get("_musicxml_path"):
        kind = ""
        ident_preview = result.get("identification") or {}
        if result.get("score_source") == "catalog":
            kind = "  [catalog edition — open this in MuseScore]"
        print(f"  musicxml → {result['_musicxml_path']}{kind}")
    if result.get("_transcribed_musicxml_path"):
        print(
            f"  transcribed musicxml → {result['_transcribed_musicxml_path']}  "
            f"[performance dump; not the PD score]"
        )
    if result.get("_midi_path"):
        print(f"  midi → {result['_midi_path']}")

    ident = result.get("identification")
    if ident:
        if ident.get("matched"):
            print(
                f"  identify: MATCH  {ident.get('title')}  "
                f"conf={ident.get('confidence'):.2f}  "
                f"edition={ident.get('edition')}  "
                f"score_source={result.get('score_source')}  "
                f"score_notes={ident.get('score_note_count', '?')}"
            )
        else:
            print(
                f"  identify: no match ({ident.get('reason')}, "
                f"best_conf={ident.get('confidence', 0):.2f}) → "
                f"fallback={ident.get('fallback', 'audio_transcription')}  "
                f"score_source={result.get('score_source')}"
            )
            print(
                "  note: data/catalog/pieces/* are bundled PD editions "
                "(ship with the repo). They are NOT a live ID until MATCH "
                "copies one into this output dir."
            )
            likely = ident.get("likely")
            if isinstance(likely, dict) and likely.get("piece_id"):
                print(
                    f"  likely (soft, not MATCH): {likely.get('title')} "
                    f"[{likely.get('piece_id')}] soft={likely.get('soft_score')} "
                    f"— {likely.get('reason')}"
                )
        if result.get("_discovery_log"):
            print(result["_discovery_log"])

    rests = result.get("rests") or []
    for r in rests[:8]:
        dur_val = r.get("duration_value", "?")
        print(
            f"  rest  {r['onset']:6.2f}–{r['offset']:6.2f}s  "
            f"dur={r['duration']:.3f}s  {dur_val}"
        )
    if len(rests) > 8:
        print(f"  … {len(rests) - 8} more rests")

    mert = result.get("mert")
    if mert:
        shape = mert["summary"]["shape"]
        print(
            f"  MERT {mert['model_id']} on {mert['device']}: "
            f"embeddings shape={shape}"
        )
        if "embeddings_path" in mert:
            print(f"  saved embeddings → {mert['embeddings_path']}")
    else:
        print("  MERT: skipped")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cello-note-detector",
        description=(
            "Monophonic cello WAV → multi-note pyin transcription, "
            "tempo/key/rhythm quantization (notes + rests), optional "
            "Shazam-like song ID (local PD catalog first, then optional "
            "remote IMSLP/Mutopia discovery; attach published MusicXML on "
            "confident MATCH; otherwise keep the audio transcription), "
            "optional MusicXML/MIDI sheet export, and optional "
            "MERT-v1-95M frame embeddings. Tempo/key are inferred from "
            "audio (never from filenames); use --bpm / --key to override. "
            "Bundled catalog files are not a live ID until MATCH."
        ),
    )
    p.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_RECORDINGS),
        help=(
            "WAV file or directory of WAVs "
            f"(default: {DEFAULT_RECORDINGS})"
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSON / MusicXML / MIDI (and optional .npy) outputs",
    )
    p.add_argument(
        "--musicxml",
        action="store_true",
        help=(
            "Also write bass-clef MusicXML next to the JSON "
            "(requires rhythm; open in MuseScore / Dorico / Finale)"
        ),
    )
    p.add_argument(
        "--midi",
        action="store_true",
        help="Also write a Type-1 MIDI file next to the JSON (requires rhythm)",
    )
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face MERT model id (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    p.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Also write full MERT embeddings as .npy next to the JSON",
    )
    p.add_argument(
        "--skip-mert",
        action="store_true",
        help="Only run pyin + rhythm (no model download)",
    )
    p.add_argument(
        "--skip-rhythm",
        action="store_true",
        help="Disable tempo estimation and duration/rest quantization",
    )
    p.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Override estimated tempo (BPM). Never read from filenames.",
    )
    p.add_argument(
        "--key",
        default=None,
        help=(
            "Override inferred key signature (e.g. Gm, Bb, D major, A minor). "
            "Written into JSON + MusicXML. Never read from filenames."
        ),
    )
    p.add_argument(
        "--quantize",
        choices=("coarse", "medium", "fine", "score"),
        default=DEFAULT_QUANTIZE,
        help=(
            "Rhythm coarseness preset: coarse (quarters/eighths), "
            "medium (default; +sixteenths), fine (incl. 32nds), "
            "score (equal-stream / written-intent reconstruction). "
            f"Default: {DEFAULT_QUANTIZE}"
        ),
    )
    p.add_argument(
        "--grid-beats",
        type=float,
        default=None,
        help=(
            "Override beat subdivision for onset snap "
            f"(preset default for medium: {DEFAULT_GRID_BEATS} = eighth)"
        ),
    )
    p.add_argument(
        "--min-rest-sec",
        type=float,
        default=None,
        help=(
            "Minimum silence gap (seconds) to emit as a rest "
            f"(preset default for medium: {DEFAULT_MIN_REST_SEC})"
        ),
    )
    p.add_argument(
        "--no-edge-rests",
        action="store_true",
        help="Omit leading/trailing rests; keep only inter-note gaps",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help=(
            "Drop note events shorter than this many seconds "
            f"(default: {DEFAULT_MIN_DURATION})"
        ),
    )
    p.add_argument(
        "--gap-tolerance",
        type=float,
        default=DEFAULT_GAP_TOLERANCE,
        help=(
            "Max unvoiced gap (seconds) to bridge inside one note "
            f"(default: {DEFAULT_GAP_TOLERANCE})"
        ),
    )
    p.add_argument(
        "--median-filter-size",
        type=int,
        default=DEFAULT_MEDIAN_FILTER_SIZE,
        help=(
            "Odd-ish median window over voiced MIDI frames to reduce flicker "
            f"(default: {DEFAULT_MEDIAN_FILTER_SIZE}; 1 disables)"
        ),
    )
    p.add_argument(
        "--no-onsets",
        action="store_true",
        help="Disable onset-based splits (pitch-change / gaps only)",
    )
    p.add_argument(
        "--identify",
        action="store_true",
        help=(
            "Shazam-like song ID: fingerprint audio → search the local "
            "bundled PD catalog → on miss, optionally discover remote PD "
            "MusicXML (see --discover). Implies --score-source auto "
            "unless --score-source is set. Bundled data/catalog/pieces/* "
            "are fixtures until a MATCH copies them into outputs/."
        ),
    )
    p.add_argument(
        "--discover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --identify: after a local catalog miss, try remote PD "
            "seeds in data/catalog/discovery_index.json (IMSLP/Mutopia/"
            "direct MusicXML URLs). Default: on. Use --no-discover for "
            "local-only."
        ),
    )
    p.add_argument(
        "--discovery-index",
        default=str(DEFAULT_DISCOVERY_PATH),
        help=(
            "Remote PD discovery seed JSON "
            f"(default: {DEFAULT_DISCOVERY_PATH})"
        ),
    )
    p.add_argument(
        "--score-source",
        choices=SCORE_SOURCES,
        default=None,
        help=(
            "Where sheet MusicXML/MIDI come from: "
            "auto (catalog on confident match, else transcription), "
            "transcribe (always audio transcription; default without "
            "--identify), catalog (prefer published PD score; still "
            "falls back to transcription if unmatched — never fails). "
            "Default: auto with --identify, else transcribe."
        ),
    )
    p.add_argument(
        "--catalog",
        default=str(DEFAULT_CATALOG_PATH),
        help=(
            "Path to local catalog JSON "
            f"(default: {DEFAULT_CATALOG_PATH})"
        ),
    )
    p.add_argument(
        "--list-catalog",
        action="store_true",
        help=(
            "Print every local PD catalog entry and exit "
            "(no audio processing). Use --search-catalog Q to filter."
        ),
    )
    p.add_argument(
        "--search-catalog",
        metavar="QUERY",
        default=None,
        help=(
            "PD lookup: filter catalog by id/title/composer/edition/IMSLP "
            "substring and exit. Empty catalog → transcription fallback on "
            "identify runs; this flag only lists matches."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_catalog or args.search_catalog is not None:
        query = args.search_catalog if args.search_catalog is not None else ""
        hits = search_catalog(query, args.catalog)
        print(f"Catalog ({len(hits)} hit(s)) from {args.catalog}")
        print(format_catalog_listing(hits))
        return 0

    wavs = _discover_wavs(Path(args.path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_source = _resolve_score_source(
        score_source=args.score_source,
        identify=args.identify,
    )
    want_identify = args.identify or score_source in ("auto", "catalog")

    try:
        quantize_cfg = resolve_quantize_preset(
            args.quantize,
            grid_beats=args.grid_beats,
            min_rest_sec=args.min_rest_sec,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    catalog_entries = None
    if want_identify:
        catalog_entries = load_catalog(args.catalog)
        seeds = load_discovery_index(args.discovery_index) if args.discover else []
        print(
            f"Catalog: {len(catalog_entries)} bundled piece(s) from {args.catalog} "
            f"(score-source={score_source}; "
            f"discover={'on' if args.discover else 'off'}, "
            f"{len(seeds)} remote seed(s))"
        )
        print(
            "  UX: bundled data/catalog/pieces/* ship with the repo; "
            "MATCH only after identify copies an edition into -o."
        )
        if not catalog_entries:
            print(
                "  warning: empty catalog — will try discovery (if on) then "
                "fall back to audio transcription"
            )

    mert_bundle = None
    if not args.skip_mert:
        print(f"Loading {args.model_id} …")
        mert_bundle = load_mert(args.model_id, device=args.device)
        print(f"MERT ready on {mert_bundle.device} (hidden={mert_bundle.hidden_size})")

    print(f"Processing {len(wavs)} file(s) → {output_dir}")
    for wav in wavs:
        result = _process_one(
            wav,
            output_dir=output_dir,
            mert_bundle=mert_bundle,
            save_embeddings=args.save_embeddings,
            skip_mert=args.skip_mert,
            skip_rhythm=args.skip_rhythm,
            min_duration=args.min_duration,
            gap_tolerance=args.gap_tolerance,
            median_filter_size=args.median_filter_size,
            use_onsets=not args.no_onsets,
            quantize_cfg=quantize_cfg,
            bpm_override=args.bpm,
            key_override=args.key,
            no_edge_rests=args.no_edge_rests,
            export_musicxml=args.musicxml,
            export_midi=args.midi,
            identify=args.identify,
            score_source=score_source,
            catalog_entries=catalog_entries,
            discover=bool(args.discover),
            discovery_path=args.discovery_index,
        )
        _print_summary(result)

    extras = []
    if args.musicxml:
        extras.append("MusicXML")
    if args.midi:
        extras.append("MIDI")
    extra_bit = f" (+{'/'.join(extras)})" if extras else ""
    print(f"\nDone. Wrote {len(wavs)} JSON file(s){extra_bit} under {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
