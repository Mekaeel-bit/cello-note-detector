#!/usr/bin/env python3
"""Register any PD MusicXML/MXL into the local extensible catalog.

Examples:
  python scripts/add_catalog_piece.py \\
      --id bwv1007_prelude \\
      --musicxml path/to/edition.musicxml \\
      --title "Bach Cello Suite No. 1 — Prélude" \\
      --composer "Johann Sebastian Bach" \\
      --license "Public Domain" \\
      --edition "Mutopia/Schirmer 1916"

  python scripts/add_catalog_piece.py \\
      --id gabrielli_ricercari \\
      --musicxml path/to/IMSLP923358.mxl \\
      --title "Gabrielli 7 Ricercari" \\
      --composer "Domenico Gabrielli" \\
      --license "CC0 1.0" \\
      --imslp "https://imslp.org/wiki/Ricercari,_canone_e_sonate_per_violoncello_(Gabrielli,_Domenico)"

Copies the edition file into data/catalog/pieces/<id>/edition.(musicxml|mxl),
writes fingerprint.json (pitches + ornament stats), and upserts catalog.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cello_note_detector.identify.musicxml_edition import (  # noqa: E402
    parse_musicxml_fingerprint,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", required=True, help="Catalog piece id (kebab-case)")
    p.add_argument("--musicxml", required=True, type=Path, help="PD .musicxml or .mxl")
    p.add_argument("--title", required=True)
    p.add_argument("--composer", required=True)
    p.add_argument("--edition", default="")
    p.add_argument("--license", default="Public Domain")
    p.add_argument("--imslp", default=None)
    p.add_argument("--key", default=None)
    p.add_argument("--min-confidence", type=float, default=0.55)
    p.add_argument(
        "--typical-duration-sec",
        type=float,
        default=None,
        help="Optional duration hint for multi-movement selection",
    )
    args = p.parse_args()

    src = args.musicxml.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"MusicXML/MXL not found: {src}")

    piece_dir = _REPO / "data" / "catalog" / "pieces" / args.id
    piece_dir.mkdir(parents=True, exist_ok=True)
    dest_name = "edition" + src.suffix.lower()
    if dest_name not in {"edition.musicxml", "edition.mxl", "edition.xml"}:
        dest_name = "edition.musicxml"
    dest = piece_dir / dest_name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)

    fp = parse_musicxml_fingerprint(dest)
    fp_path = piece_dir / "fingerprint.json"
    fp_path.write_text(json.dumps(fp, indent=2) + "\n", encoding="utf-8")

    meta = {
        "id": args.id,
        "title": args.title,
        "composer": args.composer,
        "edition": args.edition or args.license,
        "license": args.license,
        "imslp_work": args.imslp,
        "key": args.key,
        "min_confidence": args.min_confidence,
        "edition_musicxml": f"data/catalog/pieces/{args.id}/{dest.name}",
        "fingerprint_json": f"data/catalog/pieces/{args.id}/fingerprint.json",
        "typical_duration_sec": args.typical_duration_sec,
        "ornaments": fp.get("ornaments"),
        "num_notes": fp.get("num_notes"),
        "num_measures": fp.get("num_measures"),
    }
    (piece_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    catalog_path = _REPO / "data" / "catalog" / "catalog.json"
    if catalog_path.is_file():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    else:
        catalog = {
            "version": 2,
            "description": (
                "Extensible local catalog of public-domain cello scores "
                "(IMSLP/Mutopia/CC0 MusicXML). Add pieces with "
                "scripts/add_catalog_piece.py."
            ),
            "default_min_confidence": 0.55,
            "entries": [],
        }
    entries = catalog.setdefault("entries", [])
    entries = [e for e in entries if e.get("id") != args.id]
    entries.append(
        {
            "id": args.id,
            "title": args.title,
            "composer": args.composer,
            "edition": meta["edition"],
            "license": args.license,
            "imslp_work": args.imslp,
            "key": args.key,
            "min_confidence": args.min_confidence,
            "edition_musicxml": meta["edition_musicxml"],
            "fingerprint_json": meta["fingerprint_json"],
            "typical_duration_sec": args.typical_duration_sec,
            "match_notes": (
                "Fingerprint extracted from the edition MusicXML "
                "(monophonic pitch sequence). Slurs/trills preserved in the "
                "verbatim edition file."
            ),
        }
    )
    catalog["entries"] = entries
    catalog["version"] = max(2, int(catalog.get("version", 2)))
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(
        f"Registered {args.id}: notes={fp['num_notes']} "
        f"measures={fp['num_measures']} ornaments={fp['ornaments']} → {dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
